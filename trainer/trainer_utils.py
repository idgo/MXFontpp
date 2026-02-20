"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn

import numpy as np
from scipy.optimize import linear_sum_assignment


def cyclize(loader):
    """ Cyclize loader """
    while True:
        for x in loader:
            yield x


def has_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True

    return False


def unflatten_B(t):
    """ Unflatten [B*3, ...] tensor to [B, 3, ...] tensor
    t is flattened tensor from component batch, which is [B, 3, ...] tensor
    """
    shape = t.shape
    return t.view(shape[0]//3, 3, *shape[1:])


def overwrite_weight(model, pre_weight, verbose=False):
    """Load weights from pre_weight into model, skipping mismatched shapes."""
    model_dict = model.state_dict()
    
    # Filter out weights that don't exist or have mismatched shapes
    filtered_weight = {}
    skipped_keys = []
    for k, v in pre_weight.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_weight[k] = v
            else:
                skipped_keys.append((k, v.shape, model_dict[k].shape))
    
    if verbose and skipped_keys:
        print(f"Skipped {len(skipped_keys)} weights due to shape mismatch:")
        for k, ckpt_shape, model_shape in skipped_keys:
            print(f"  {k}: checkpoint {ckpt_shape} vs model {model_shape}")
    
    model_dict.update(filtered_weight)
    model.load_state_dict(model_dict)
    
    return skipped_keys


def load_checkpoint(path, gen, disc, aux_clf, g_optim=None, d_optim=None, ac_optim=None, force_overwrite=False):
    ckpt = torch.load(path, weights_only=False)

    # Load generator
    if force_overwrite:
        overwrite_weight(gen, ckpt['generator'], verbose=True)
    else:
        skipped = overwrite_weight(gen, ckpt['generator'], verbose=True)
        if not skipped and g_optim is not None:
            # No shape mismatches, try loading optimizer
            try:
                g_optim.load_state_dict(ckpt['optimizer'])
            except (ValueError, KeyError) as e:
                print(f"WARNING: Could not load generator optimizer state: {e}")
        elif skipped:
            print("Generator optimizer state reset due to weight mismatches.")

    # Load discriminator
    if disc is not None and 'discriminator' in ckpt:
        if force_overwrite:
            overwrite_weight(disc, ckpt['discriminator'], verbose=True)
        else:
            skipped = overwrite_weight(disc, ckpt['discriminator'], verbose=True)
            if not skipped and d_optim is not None:
                try:
                    d_optim.load_state_dict(ckpt['d_optimizer'])
                except (ValueError, KeyError) as e:
                    print(f"WARNING: Could not load discriminator optimizer state: {e}")
            elif skipped:
                print("Discriminator optimizer state reset due to weight mismatches.")

    # Load auxiliary classifier
    if aux_clf is not None and 'aux_clf' in ckpt:
        if force_overwrite:
            overwrite_weight(aux_clf, ckpt['aux_clf'], verbose=True)
        else:
            skipped = overwrite_weight(aux_clf, ckpt['aux_clf'], verbose=True)
            if not skipped and ac_optim is not None:
                try:
                    ac_optim.load_state_dict(ckpt['ac_optimizer'])
                except (ValueError, KeyError) as e:
                    print(f"WARNING: Could not load aux_clf optimizer state: {e}")
            elif skipped:
                print("Auxiliary classifier optimizer state reset due to weight mismatches.")
            
    st_epoch = ckpt.get('epoch', 0)
    if force_overwrite:
        st_epoch = 0
    loss = ckpt.get('loss', 0.0)

    return st_epoch, loss


def binarize_labels(label_ids, n_labels):
    binary_labels = []
    for _lids in label_ids:
        _blabel = torch.eye(n_labels)[_lids].sum(0).bool()
        binary_labels.append(_blabel)
    binary_labels = torch.stack(binary_labels)

    return binary_labels


def expert_assign(prob_org):
    n_comp, n_exp = prob_org.shape
    neg_prob = -prob_org.T if n_comp < n_exp else -prob_org
    n_row, n_col = neg_prob.shape

    prob_in = neg_prob
    remain_rs = np.arange(n_row)
    selected_rs = []
    selected_cs = []

    while len(remain_rs):
        r_in, c_in = linear_sum_assignment(prob_in)
        r_org = remain_rs[r_in]
        selected_rs.append(r_org)
        selected_cs.append(c_in)
        remain_rs = np.delete(remain_rs, r_in)
        prob_in = neg_prob[remain_rs]

    cat_selected_rs = np.concatenate(selected_cs) if n_comp < n_exp else np.concatenate(selected_rs)
    cat_selected_cs = np.concatenate(selected_rs) if n_comp < n_exp else np.concatenate(selected_cs)

    cat_selected_rs = torch.LongTensor(cat_selected_rs).cuda()
    cat_selected_cs = torch.LongTensor(cat_selected_cs).cuda()

    return cat_selected_rs, cat_selected_cs
