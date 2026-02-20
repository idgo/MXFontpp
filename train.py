"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import json
import sys
from pathlib import Path
import argparse
import random

import torch

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torchvision import transforms
import numpy as np
from sconf import Config, dump_args
import utils
from utils import Logger

from models import Generator, disc_builder, aux_clf_builder
from models.modules import weights_init
from trainer import FactTrainer, Evaluator, load_checkpoint
from datasets import get_trn_loader, get_val_loader, get_image_trn_loader, get_image_val_loader


def setup_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")
    parser.add_argument("--fixed_char_txt", type=str, default=None,
                        help="Path to a text file containing the characters for training")

    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)

    if cfg.use_ddp:
        cfg.n_workers = 0

    cfg.work_dir = Path(cfg.work_dir)
    (cfg.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    return args, cfg


def setup_transforms(cfg):
    if cfg.dset_aug.random_affine:
        aug_transform = [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=10, translate=(0.03, 0.03), scale=(0.9, 1.1), shear=10, fillcolor=255
            )
        ]
    else:
        aug_transform = []

    tensorize_transform = [transforms.Resize((128, 128)), transforms.ToTensor()]
    if cfg.dset_aug.normalize:
        tensorize_transform.append(transforms.Normalize([0.5], [0.5]))
        cfg.g_args.dec.out = "tanh"

    trn_transform = transforms.Compose(aug_transform + tensorize_transform)
    val_transform = transforms.Compose(tensorize_transform)

    return trn_transform, val_transform


def cleanup():
    dist.destroy_process_group()


def is_main_worker(gpu):
    return (gpu <= 0)


def train_ddp(gpu, args, cfg, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:" + str(cfg.port),
        world_size=world_size,
        rank=gpu,
    )
    cfg.batch_size = cfg.batch_size // world_size
    train(args, cfg, ddp_gpu=gpu)
    cleanup()


def train(args, cfg, ddp_gpu=-1):
    cfg.gpu = ddp_gpu
    torch.cuda.set_device(ddp_gpu)
    cudnn.benchmark = True

    logger_path = cfg.work_dir / "log.log"
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    image_scale = 0.5
    image_path = cfg.work_dir / "images"
    writer = utils.DiskWriter(image_path, scale=image_scale)
    cfg.tb_freq = -1

    args_str = dump_args(args)
    if is_main_worker(ddp_gpu):
        logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
        logger.info("Args:\n{}".format(args_str))
        logger.info("Configs:\n{}".format(cfg.dumps()))

    logger.info("Get dataset ...")

    trn_transform, val_transform = setup_transforms(cfg)

    primals = json.load(open(cfg.primals))
    decomposition = json.load(open(cfg.decomposition))
    n_comps = len(primals)

    if args.fixed_char_txt:
        with open(args.fixed_char_txt, "r") as f:
            fixed_chars = f.read().strip()
        char_filter = list(set(decomposition).intersection(fixed_chars))
        logger.info(f"Using fixed {len(char_filter)} chars for training")
    else:
        char_filter = list(decomposition)

    # Select dataset type based on config (default: "ttf" for backward compatibility)
    dataset_type = cfg.get("dataset_type", "ttf")
    logger.info(f"Using dataset type: {dataset_type}")
    
    if dataset_type == "image":
        # Use image folder dataset
        trn_dset, trn_loader = get_image_trn_loader(cfg.dset.train,
                                              primals,
                                              decomposition,
                                              trn_transform,
                                              char_filter=char_filter,
                                              use_ddp=cfg.use_ddp,
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.n_workers,
                                              shuffle=True)
        
        test_dset, test_loader = get_image_val_loader(cfg.dset.val,
                                                val_transform,
                                                batch_size=cfg.batch_size,
                                                num_workers=cfg.n_workers,
                                                shuffle=False)
    else:
        # Use TTF dataset (default)
        trn_dset, trn_loader = get_trn_loader(cfg.dset.train,
                                              primals,
                                              decomposition,
                                              trn_transform,
                                              char_filter=char_filter,
                                              use_ddp=cfg.use_ddp,
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.n_workers,
                                              shuffle=True)

        test_dset, test_loader = get_val_loader(cfg.dset.val,
                                                val_transform,
                                                batch_size=cfg.batch_size,
                                                num_workers=cfg.n_workers,
                                                shuffle=False)

    logger.info("Build model ...")

    # Fine-tuning config
    ft_cfg = cfg.get("fine_tune", {})
    is_fine_tune = ft_cfg.get("enabled", False)
    if is_fine_tune:
        logger.info("=" * 60)
        logger.info("FINE-TUNING MODE")
        logger.info("=" * 60)

    # generator
    g_kwargs = cfg.get("g_args", {})
    gen = Generator(1, cfg.C, 1, **g_kwargs)
    gen.cuda()
    gen.apply(weights_init(cfg.init))

    d_kwargs = cfg.get("d_args", {})
    disc = disc_builder(cfg.C, trn_dset.n_fonts, trn_dset.n_chars, **d_kwargs)
    disc.cuda()
    disc.apply(weights_init(cfg.init))

    aux_clf = aux_clf_builder(gen.feat_shape["last"], trn_dset.n_fonts, n_comps, **cfg.ac_args)
    aux_clf.cuda()
    aux_clf.apply(weights_init(cfg.init))

    # Load checkpoint (must happen before freezing so we load pretrained weights first)
    st_step = 0
    if cfg.resume:
        # For fine-tuning, force_resume=True ensures:
        # - Mismatched layers (disc/AC embeddings) are skipped gracefully
        # - Step counter resets to 0
        # - Optimizer state is fresh (not loaded from checkpoint)
        force = cfg.force_resume or is_fine_tune
        st_step, loss = load_checkpoint(cfg.resume, gen, disc, aux_clf, g_optim=None, d_optim=None, ac_optim=None, force_overwrite=force)
        logger.info("Loaded checkpoint from {} (Step {}, Loss {:7.3f})".format(cfg.resume, st_step, loss))
        if is_fine_tune:
            st_step = 0
            logger.info("Fine-tuning: reset step to 0, fresh optimizer state.")

    # Fine-tuning: freeze layers as configured
    if is_fine_tune:
        if ft_cfg.get("freeze_style_enc", False):
            logger.info("Freezing: Style Encoder")
            for p in gen.style_enc.parameters():
                p.requires_grad = False

        if ft_cfg.get("freeze_experts", False):
            logger.info("Freezing: Expert Networks")
            for p in gen.experts.parameters():
                p.requires_grad = False

        if ft_cfg.get("freeze_fact_blocks", False):
            logger.info("Freezing: Factorization Blocks")
            for p in gen.fact_blocks.parameters():
                p.requires_grad = False
            for p in gen.recon_blocks.parameters():
                p.requires_grad = False

        # Log trainable parameter counts
        total_params = sum(p.numel() for p in gen.parameters())
        trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        logger.info(f"Generator: {trainable_params:,} / {total_params:,} trainable params "
                    f"({100*trainable_params/total_params:.1f}%)")

    # Create optimizers (after freezing, so only trainable params get optimizer state)
    g_trainable = filter(lambda p: p.requires_grad, gen.parameters())
    g_optim = optim.Adam(g_trainable, lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas)
    ac_optim = optim.Adam(aux_clf.parameters(), lr=cfg.ac_lr, betas=cfg.adam_betas)

    # For non-fine-tune resume, try to load optimizer states
    if cfg.resume and not is_fine_tune and not cfg.force_resume:
        try:
            ckpt = torch.load(cfg.resume, weights_only=False)
            g_optim.load_state_dict(ckpt['optimizer'])
            if 'd_optimizer' in ckpt:
                d_optim.load_state_dict(ckpt['d_optimizer'])
            if 'ac_optimizer' in ckpt:
                ac_optim.load_state_dict(ckpt['ac_optimizer'])
            logger.info("Loaded optimizer states from checkpoint.")
        except Exception as e:
            logger.warning(f"Could not load optimizer states: {e}")

    evaluator = Evaluator(writer)

    trainer = FactTrainer(gen, disc, g_optim, d_optim,
                          aux_clf, ac_optim,
                          writer, logger,
                          evaluator, test_loader,
                          cfg)

    trainer.train(trn_loader, st_step, cfg.max_iter)


def main():
    args, cfg = setup_args_and_config()

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    if cfg.use_ddp:
        ngpus_per_node = torch.cuda.device_count()
        world_size = ngpus_per_node
        mp.spawn(train_ddp, nprocs=ngpus_per_node, args=(args, cfg, world_size))
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()
