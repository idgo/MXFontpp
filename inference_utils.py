import copy
from pathlib import Path
from PIL import Image

import torch
from sconf import Config
from torchvision import transforms

import models
from datasets import render
from utils import refine

def find_reference_image(ref_dir, char, all_ref_paths, index, total):
    """
    Finds the corresponding reference image for a character.
    """
    supported_extensions = ["png", "jpg", "jpeg"]
    for ext in supported_extensions:
        potential_ref_path = Path(ref_dir) / f"{char}.{ext}"
        if potential_ref_path.exists():
            return str(potential_ref_path)

    # Fallback to finding any file with the character name
    for ext in supported_extensions:
        char_files = list(Path(ref_dir).glob(f"*{char}.{ext}"))
        if char_files:
            return str(char_files[0])
        
    # Fallback to using an image from the list by index
    if all_ref_paths:
        return str(all_ref_paths[index % len(all_ref_paths)])
    
    return "" # Return empty if no image found


class FontGenerator:
    """Encapsulates the font generation logic."""
    def __init__(self, weight_path, cfg_path="cfgs/eval.yaml", base_weight_path=None):
        """
        Initializes the FontGenerator and loads the model.

        Automatically detects and handles three checkpoint formats:
          1. Standard checkpoint  (e.g. 340000.pth, merged.pth)
          2. Full LoRA checkpoint (e.g. last.pth from finetune_lora.py)
          3. LoRA-only weights    (e.g. lora_weights.pth)

        Args:
            weight_path:      Path to the model checkpoint.
            cfg_path:         Path to the model config YAML.
            base_weight_path: (Optional) Path to the base pretrained checkpoint.
                              Required when weight_path is a LoRA-only weights file
                              and the embedded path is not accessible.
        """
        print("Initializing FontGenerator...")
        self.model, self.cfg = self._load_model(weight_path, cfg_path, base_weight_path)
        print("FontGenerator initialized.")

    @staticmethod
    def _detect_lora_config(state_dict):
        """
        Detect LoRA parameters in a state dict and infer the config.

        Returns None if no LoRA keys found, otherwise returns a dict with
        rank, alpha, dropout, and target module suffixes.
        """
        lora_a_keys = [k for k in state_dict if '.lora_A.' in k]
        if not lora_a_keys:
            return None

        # Infer rank from the first lora_A weight (out_channels = rank)
        rank = state_dict[lora_a_keys[0]].shape[0]

        # Infer target suffixes from the module paths
        # e.g. "experts.experts.0.layers.5.attn.qkv.lora_A.weight" → "attn.qkv"
        targets = set()
        for k in lora_a_keys:
            module_path = k.rsplit('.lora_A.', 1)[0]  # strip ".lora_A.weight"
            parts = module_path.split('.')
            suffix = '.'.join(parts[-2:])  # last two segments
            targets.add(suffix)

        return {
            'rank': rank,
            'alpha': float(rank),
            'dropout': 0.0,
            'targets': sorted(targets),
        }

    @staticmethod
    def _infer_n_experts_from_state_dict(state_dict):
        """Infer number of experts from checkpoint keys (experts.experts.0, ..., experts.experts.N)."""
        import re
        max_expert = -1
        for key in state_dict:
            m = re.match(r"experts\.experts\.(\d+)\.", key)
            if m:
                max_expert = max(max_expert, int(m.group(1)))
        return (max_expert + 1) if max_expert >= 0 else None

    def _load_model(self, weight_path, cfg_path="cfgs/eval.yaml", base_weight_path=None):
        """
        Builds and loads the trained MX-Font model.

        Handles standard, full-LoRA, and LoRA-only checkpoint formats.
        Infers n_experts from checkpoint when it does not match the config (e.g. 8-experts checkpoint).
        """
        print(f"Loading model from {weight_path} ...")
        ckpt = torch.load(weight_path, map_location=torch.device('cuda'), weights_only=False)

        # Determine state_dict to use for architecture inference and loading
        if 'lora_state_dict' in ckpt:
            base_path = base_weight_path or ckpt.get('pretrained_checkpoint')
            if not base_path or not Path(base_path).exists():
                raise FileNotFoundError(
                    f"LoRA-only weights require a base checkpoint. "
                    f"Embedded path '{ckpt.get('pretrained_checkpoint')}' not found. "
                    f"Pass base_weight_path= to FontGenerator."
                )
            base_ckpt = torch.load(base_path, map_location='cuda', weights_only=False)
            state_for_infer = base_ckpt.get('generator_ema', base_ckpt.get('generator', base_ckpt))
        else:
            state_for_infer = ckpt.get('generator_ema', ckpt.get('generator', ckpt))

        # Infer n_experts from checkpoint so architecture matches (e.g. 8-experts vs default 6)
        inferred_n_experts = self._infer_n_experts_from_state_dict(state_for_infer)
        cfg = Config(cfg_path, default="cfgs/defaults.yaml")
        g_kwargs = copy.deepcopy(cfg.get('g_args', {}))
        if inferred_n_experts is not None:
            config_n_experts = g_kwargs.get('experts', {}).get('n_experts')
            if config_n_experts != inferred_n_experts:
                print(f"  Checkpoint has {inferred_n_experts} experts; using that instead of config ({config_n_experts}).")
                if 'experts' not in g_kwargs:
                    g_kwargs['experts'] = {}
                g_kwargs['experts'] = copy.deepcopy(g_kwargs['experts'])
                g_kwargs['experts']['n_experts'] = inferred_n_experts
        model = models.Generator(1, cfg.C, 1, **g_kwargs).cuda()

        # ── Case 3: LoRA-only weights file (lora_weights.pth) ──────────
        if 'lora_state_dict' in ckpt:
            lora_cfg = ckpt['lora_config']
            lora_state = ckpt['lora_state_dict']
            base_path = base_weight_path or ckpt.get('pretrained_checkpoint')
            print(f"  Loading base model from {base_path}")
            model.load_state_dict(state_for_infer)

            print(f"  Injecting LoRA (rank={lora_cfg['rank']}, targets={lora_cfg['targets']})")
            inject_lora(model, lora_cfg['targets'], lora_cfg['rank'],
                        lora_cfg['alpha'], lora_cfg.get('dropout', 0.0))
            load_lora_state_dict(model, lora_state)
            merge_and_unload(model)
            print("  Merged LoRA weights into model.")

        else:
            lora_config = self._detect_lora_config(state_for_infer)

            if lora_config:
                # ── Case 2: Full LoRA checkpoint (last.pth) ────────────
                print(f"  Detected LoRA checkpoint (rank={lora_config['rank']}, "
                      f"targets={lora_config['targets']})")
                inject_lora(model, lora_config['targets'], lora_config['rank'],
                            lora_config['alpha'], lora_config.get('dropout', 0.0))
                model.load_state_dict(state_for_infer)
                merge_and_unload(model)
                print("  Merged LoRA weights into model.")
            else:
                # ── Case 1: Standard checkpoint ────────────────────────
                model.load_state_dict(state_for_infer)

        model.eval()
        print("Model loaded successfully.")
        return model, cfg

    def get_style_factors(self, ref_path, batch_size=3):
        """
        Extracts style factors from a directory of reference images.
        """
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        supported_extensions = ["png", "jpg", "jpeg"]
        ref_paths = []
        for ext in supported_extensions:
            ref_paths.extend(Path(ref_path).glob(f"*.{ext}"))
        
        if not ref_paths:
            print(f"Warning: No reference images with supported extensions ({', '.join(supported_extensions)}) found in {ref_path}")
            return None, []

        ref_imgs = torch.stack([transform(Image.open(str(p))) for p in ref_paths]).cuda()
        ref_batches = torch.split(ref_imgs, batch_size)
        
        style_facts = {}
        with torch.no_grad():
            for batch in ref_batches:
                style_fact = self.model.factorize(self.model.encode(batch), 0)
                for k in style_fact:
                    style_facts.setdefault(k, []).append(style_fact[k])
                
        style_facts = {k: torch.cat(v).mean(0, keepdim=True) for k, v in style_facts.items()}
        return style_facts, ref_paths

    def generate_batched_char_images(self, style_facts, source_font, chars, transform):
        """
        Generates character images in a batch for a single font.
        """
        source_imgs = torch.stack([transform(render(source_font, c)) for c in chars]).cuda()
        
        num_chars = len(chars)
        style_facts_replicated = {
            k: v.repeat(num_chars, *([1] * (v.dim() - 1))) for k, v in style_facts.items()
        }

        with torch.no_grad():
            char_facts = self.model.factorize(self.model.encode(source_imgs), 1)
            gen_feats = self.model.defactorize([style_facts_replicated, char_facts])
            generated_imgs = self.model.decode(gen_feats).detach().cpu()
        
        return source_imgs.cpu(), refine(generated_imgs)
