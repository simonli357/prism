#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, io, json, argparse
from collections import defaultdict, deque
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='numpy')

import time, platform, sys, json, hashlib
from datetime import datetime
try:
    from .unet_conv_lstm import UNetConvLSTM
    from .unet_attention import AttentionUNetConvLSTM
    from .shard_utils import resolve_shards, SeqFromWDS
    from .deeplabv3plus import DeepLabV3Plus
    from .cnn import CNNCorrectionNetSingle
    from .plotter import _read_training_log_csv, _plot_metric, _save_training_plots
    from .evaluate3 import run_inference
    from .train_utils import *
    from .color_mapping import color28_to_onehot14, color28_to_color14, onehot14_to_color, SEM_CHANNELS, NEW_CLASSES
except ImportError:
    from unet_conv_lstm import UNetConvLSTM
    from unet_attention import AttentionUNetConvLSTM
    from shard_utils import resolve_shards, SeqFromWDS
    from deeplabv3plus import DeepLabV3Plus
    from cnn import CNNCorrectionNetSingle
    from plotter import _read_training_log_csv, _plot_metric, _save_training_plots
    from evaluate3 import run_inference
    from color_mapping import color28_to_onehot14, color28_to_color14, onehot14_to_color, SEM_CHANNELS, NEW_CLASSES

import random

def calculate_class_weights(shard_paths, seq_len, num_workers=4):
    print(f"Processing {len(shard_paths)} shards to calculate class weights...")
    
    ds = SeqFromWDS(
        shard_paths=shard_paths,
        seq_len=seq_len,
        shuffle_shards=False,
        include_mask=False
    )
    
    loader = torch.utils.data.DataLoader(
        ds, 
        batch_size=1,
        num_workers=num_workers
    )

    class_counts = torch.zeros(SEM_CHANNELS, dtype=torch.int64)
    
    pbar = tqdm(loader, desc="Counting class pixels", ncols=100)
    for _, (Y_sem, _), _ in pbar:
        y_sem_single = Y_sem.squeeze(0)
        valid_mask = (y_sem_single > 0)
        valid_labels = y_sem_single[valid_mask]

        if valid_labels.numel() > 0:
            counts = torch.bincount(valid_labels, minlength=SEM_CHANNELS)
            class_counts += counts.cpu()

    print("\n--- Raw Pixel Counts (excluding 'Unlabeled' from total) ---")
    for i, count in enumerate(class_counts):
        print(f"Class {i:02d} ({NEW_CLASSES[i]}): {count.item():,}")

    weights = torch.zeros(SEM_CHANNELS, dtype=torch.float32)
    
    valid_counts = class_counts[1:]
    present_classes_mask = valid_counts > 0
    
    if present_classes_mask.any():
        log_weights = 1.0 / torch.log(1.02 + valid_counts[present_classes_mask])
        log_weights = (log_weights / log_weights.mean())
        weights[1:][present_classes_mask] = log_weights
    
    boost_factor_person = 5.0 
    boost_factor_water = 1.5
    boost_factor_vehicle = 3.0
    if weights[8] > 0:
        weights[8] *= boost_factor_person
    if weights[11] > 0:
        weights[11] *= boost_factor_water
    if weights[10] > 0:
        weights[10] *= boost_factor_vehicle
    for i in range(1, SEM_CHANNELS):
        if class_counts[i] == 0:
            print(f"-> WARNING: Class {i} ({NEW_CLASSES[i]}) has 0 pixels. Its weight is set to 0.")

    print("\n--- Calculated Class Weights (Log-Normalized) ---")
    for i, w in enumerate(weights):
        print(f"Class {i:02d} ({NEW_CLASSES[i]}): {w.item():.4f}")
        
    return weights.tolist()

def build_model(args, C_in, C_out, device):
    if args.model == "cnn":
        # Single-frame correction CNN; accepts [B,T,C,H,W] or [B,C,H,W] and uses the last frame
        model = CNNCorrectionNetSingle(
            in_ch_per_frame=C_in,
            base=args.base,
            blocks_stage1=4,
            blocks_stage2=6,
            aspp_rates=(1, 2, 4),
            use_identity_correction=True,
            return_edit=False,
        )
    elif args.model == "unet":
        model = UNetConvLSTM(in_ch=C_in, base=args.base, out_ch=C_out)
    elif args.model == "unet_attn":
        model = AttentionUNetConvLSTM(in_ch=C_in, base=args.base, out_ch=C_out)
    elif args.model == "deeplabv3p":
        model = DeepLabV3Plus(in_ch=C_in, out_ch=C_out)
    else:
        raise ValueError(f"Unknown model {args.model}")
    print(f"[INFO] Using model: {args.model}")
    return model.to(device)
# ----------------------------
# Losses
# ----------------------------
def weighted_masked_l1(pred, target, mask, weights, eps=1e-6):
    diff = torch.abs(pred - target) * mask
    weighted_diff = diff * weights
    denom = (mask * weights).sum() + eps
    return weighted_diff.sum() / denom
def masked_l1(pred, target, mask, eps=1e-6):
    diff = torch.abs(pred - target) * mask
    denom = mask.sum() + eps
    return diff.sum() / denom
def masked_bce_with_logits(pred_logits, target_probs, mask, eps=1e-6):
    # pred_logits, target_probs: [B,14,H,W]; mask: [B,1,H,W]
    loss = F.binary_cross_entropy_with_logits(pred_logits, target_probs, reduction='none')
    loss = loss * mask
    denom = mask.sum() * pred_logits.shape[1] + eps
    return loss.sum() / denom
def masked_cross_entropy(pred_logits, target_indices, mask, eps=1e-6):
    """
    pred_logits:    [B, C, H, W] raw scores from the model
    target_indices: [B, H, W] long tensor with class indices
    mask:           [B, 1, H, W] float tensor with 1=valid
    """
    # Exclude masked-out pixels from loss calculation
    loss = F.cross_entropy(pred_logits, target_indices, reduction='none') # -> [B,H,W]
    loss = loss * mask.squeeze(1) # remove channel dim from mask
    denom = mask.sum() + eps
    return loss.sum() / denom
def focal_loss_with_logits(pred_logits, target_indices, gamma=2.0, alpha=None, ignore_index=255, eps=1e-6):
    if alpha is None:
        alpha = 0.25 # Default back to scalar if no weights are provided

    valid_mask = (target_indices != ignore_index)
    target_indices_valid = target_indices[valid_mask]

    pred_logits_flat = pred_logits.permute(0, 2, 3, 1).reshape(-1, pred_logits.shape[1])
    pred_logits_valid = pred_logits_flat[valid_mask.flatten()]

    ce_loss = F.cross_entropy(pred_logits_valid, target_indices_valid, reduction='none')
    pt = torch.exp(-ce_loss)

    if isinstance(alpha, torch.Tensor):
        alpha_t = alpha.to(device=target_indices_valid.device)[target_indices_valid]
    else:
        alpha_t = alpha

    focal_loss = alpha_t * ((1 - pt) ** gamma) * ce_loss
    denom = valid_mask.sum() + eps
    return focal_loss.sum() / denom

def _collate(batch):
    Xs, Y_sems, Y_elevs, Ms = [], [], [], []
    for X, (Y_sem, Y_elev), M in batch:
        Xs.append(torch.from_numpy(np.transpose(X, (0,3,1,2))).float())
        Y_sems.append(torch.from_numpy(Y_sem).long())
        Y_elevs.append(torch.from_numpy(np.transpose(Y_elev, (2,0,1))).float())
        Ms.append(torch.from_numpy(M[None, ...]).float())
    return torch.stack(Xs), (torch.stack(Y_sems), torch.stack(Y_elevs)), torch.stack(Ms)

def train(args):
    def _safe_save_json(path, data_dict):
        tmp = path + ".tmp"
        save_json(tmp, data_dict)
        os.replace(tmp, path)

    set_seed(args.split_seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    UNLABELED_ID = 0

    print("--- Resolving shard paths from pre-split directories ---")
    train_dir = os.path.join(args.shards, 'train')
    val_dir   = os.path.join(args.shards, 'val')
    test_dir  = os.path.join(args.shards, 'test')
    train_shards = resolve_shards(train_dir, recursive=False, only_remapped=False)
    val_shards   = resolve_shards(val_dir,   recursive=False, only_remapped=False)
    test_shards  = resolve_shards(test_dir,  recursive=False, only_remapped=False)
    num_shards = len(train_shards) + len(val_shards) + len(test_shards)

    print(f"{num_shards} shards are split into: {len(train_shards)} train, {len(val_shards)} val, {len(test_shards)} test")
    print("-" * 29)

    run_out = resolve_out_dir(args.out, args.model)
    os.makedirs(run_out, exist_ok=True)
    print(f"\n{'='*25} Starting Training {'='*25}")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"Output directory: {run_out}\n")

    # ----- class weights (optional) -----
    class_weights = None
    if args.loss_type == 'focal' and args.auto_class_weights:
        weights_path = os.path.join(run_out, "class_weights.json")
        if os.path.exists(weights_path):
            print(f"\n--- Loading existing class weights from: {weights_path} ---")
            with open(weights_path, 'r') as f:
                weights_list = json.load(f)
            class_weights = torch.tensor(weights_list, device=device, dtype=torch.float32)
            print("--- Class Weights Loaded Successfully ---\n")
        else:
            print(f"{weights_path} does not exist.")
            print("\n--- Auto-Calculating Class Weights for Focal Loss (first run) ---")
            weights_list = calculate_class_weights(train_shards, args.seq_len, args.workers)
            class_weights = torch.tensor(weights_list, device=device, dtype=torch.float32)
            print(f"--- Saving calculated weights to: {weights_path} ---")
            with open(weights_path, 'w') as f:
                json.dump(weights_list, f, indent=4)
            print("--- Class Weight Calculation Complete ---\n")

    # ----- datasets / loaders -----
    ds_train = SeqFromWDS(shard_paths=train_shards, seq_len=args.seq_len,
                          shuffle_shards=True, include_mask=args.include_mask)
    ds_val   = SeqFromWDS(shard_paths=val_shards,   seq_len=args.seq_len,
                          shuffle_shards=False, include_mask=args.include_mask)

    loader = DataLoader(
        ds_train, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
        collate_fn=_collate, persistent_workers=(args.workers > 0),
        prefetch_factor=(4 if args.workers > 0 else None), drop_last=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, num_workers=max(1, args.workers//2), pin_memory=True,
        collate_fn=_collate, persistent_workers=(args.workers//2 > 0),
        prefetch_factor=(4 if args.workers//2 > 0 else None),
    )

    # ----- model / opt / sched -----
    C_in  = SEM_CHANNELS + 1 + (1 if args.include_mask else 0)
    C_out = SEM_CHANNELS + 1
    model = build_model(args, C_in, C_out, device)

    elev_params  = [p for n,p in model.named_parameters() if n.startswith("elev_head")]
    other_params = [p for n,p in model.named_parameters() if not n.startswith("elev_head")]
    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr,       "weight_decay": 1e-4},
            {"params": elev_params,  "lr": args.lr * 0.5, "weight_decay": 3e-4},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    start_epoch = 1
    if args.resume:
        ckpt_path = args.resume
        if os.path.isdir(ckpt_path):
            # if a dir is provided, pick the latest epoch checkpoint in that dir
            maybe = _latest_epoch_ckpt(ckpt_path)
            if maybe is None:
                print(f"[resume] No epoch checkpoints found in dir: {ckpt_path}")
            else:
                ckpt_path = maybe
        if os.path.isfile(ckpt_path):
            print(f"[resume] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            # scheduler: either restore state or set last_epoch to match
            if "sched" in ckpt:
                try:
                    sched.load_state_dict(ckpt["sched"])
                except Exception as e:
                    print(f"[resume] scheduler load warning: {e}; falling back to last_epoch")
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=max(1, args.epochs), last_epoch=ckpt.get("epoch", 0)
                    )
            else:
                # keep continuity even if old ckpt didn't have a saved scheduler
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, args.epochs), last_epoch=ckpt.get("epoch", 0)
                )
            # next epoch to run
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[resume] Resuming at epoch {start_epoch}")

            # initialize best_miou from prior best checkpoint if present
            try:
                prev_best = os.path.join(resolve_out_dir(args.out, args.model), "checkpoint_best.pt")
                if os.path.exists(prev_best):
                    _b = torch.load(prev_best, map_location="cpu")
                    best_miou = float(_b["val_metrics"]["miou"])
                    print(f"[resume] Previous best mIoU found: {best_miou:.6f}")
                else:
                    best_miou = float(ckpt.get("val_metrics", {}).get("miou", -1.0))
                    print(f"[resume] Using last ckpt mIoU as best: {best_miou:.6f}")
            except Exception as e:
                print(f"[resume] Could not determine previous best mIoU: {e}")
                best_miou = -1.0

            # if user specified fewer/equal total epochs than already finished, skip training
            if start_epoch > args.epochs:
                if args.allow_new_epochs:
                    print(f"[resume] Note: already passed target epochs ({args.epochs}). "
                        f"Proceeding directly to final evaluation.")
                else:
                    print(f"[resume] Nothing to do: start_epoch ({start_epoch}) > epochs ({args.epochs}). Exiting.")
                    return
        else:
            print(f"[resume] Path does not exist: {ckpt_path} (starting fresh)")
            
    os.makedirs(args.out, exist_ok=True)
    best_miou = -1.0

    # ----- csv log header -----
    log_file_path = os.path.join(run_out, "training_log.csv")
    with open(log_file_path, 'w') as f:
        f.write("epoch,miou,pix_acc,elev_mae,elev_rmse,bf_score\n")

    # ----- losses -----
    if args.loss_type == 'focal':
        print("Using Focal Loss for semantics.")
        sem_loss_fn = lambda p, t: focal_loss_with_logits(p, t,
                                                          gamma=args.focal_gamma,
                                                          alpha=class_weights,
                                                          ignore_index=255)
    else:
        print("Using Cross Entropy Loss for semantics.")
        sem_loss_fn = lambda p, t: F.cross_entropy(p, t, ignore_index=255)

    # ----- manifest scaffold -----
    lr_hist = []
    run_start_ts = time.time()
    run_meta = {
        "type": "training",
        "start_time_utc": datetime.utcnow().isoformat() + "Z",
        "args": {k: _to_serializable(v) for k, v in vars(args).items()},
        "split_seed": args.split_seed,
        "include_mask": args.include_mask,
        "sem_channels": SEM_CHANNELS,
        "model_name": args.model,
        "output_dir": run_out,
        "dataset": {
            "root": args.shards,
            "train_shards_count": len(train_shards),
            "val_shards_count": len(val_shards),
            "test_shards_count": len(test_shards),
            "seq_len": args.seq_len,
        },
        "env": env_info(),
        "optimizer": {
            "type": "AdamW",
            "groups": [
                {"name": "other_params", "lr": args.lr, "weight_decay": 1e-4},
                {"name": "elev_head",    "lr": args.lr * 0.5, "weight_decay": 3e-4},
            ],
            "grad_clip": 1.0
        },
        "loss": {
            "sem_type": args.loss_type,
            "focal_gamma": (args.focal_gamma if args.loss_type == "focal" else None),
            "elev": {
                "space": "linear",
                "robust": "huber",
                "delta": 0.1,
                "tv_weight": 0.02,
                "object_weight_nominal": 4.0,
                "weight_normalization": "mean≈1 over valid mask"
            },
            "weights": {"w_sem": args.w_sem, "w_elev": args.w_elev}
        },
        "scheduler": {"type": "CosineAnnealingLR", "T_max": max(1, args.epochs)},
        "lr_per_epoch": [],
        "train_losses_per_epoch": [],
        "val_metrics_per_epoch": [],
        "checkpoints": {"best_by_miou": None},
        "model_params": model_num_params(model),
    }
    if class_weights is not None:
        run_meta["class_weights"] = [float(x) for x in class_weights.tolist()]

    _safe_save_json(os.path.join(run_out, "manifest.config.json"), run_meta)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_sem_sum  = 0.0
        epoch_elev_sum = 0.0
        epoch_tv_sum   = 0.0
        epoch_steps    = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs} [Training]", ncols=110)
        for step, (X, Y_tuple, M) in enumerate(pbar, start=1):
            if X.size(0) < 2:
                print("[warning] skipping batch with <2 samples")
                continue
            X = X.to(device, non_blocking=True)
            Y_sem, Y_elev = Y_tuple
            Y_sem  = Y_sem.to(device, non_blocking=True)
            Y_elev = Y_elev.to(device, non_blocking=True)
            M      = M.to(device, non_blocking=True)

            # object-weight map (normalized to mean ~1 on valid mask)
            object_ids = [3,4,5,6,7,8,9,10]
            weight_map = torch.ones_like(Y_elev, device=device)
            object_weight = 4.0
            for obj_id in object_ids:
                weight_map[Y_sem.unsqueeze(1) == obj_id] = object_weight
            with torch.no_grad():
                denom = (M * weight_map).sum().clamp_min(1.0)
                scale = (M.sum() / denom)
                weight_map = weight_map * scale

            # forward
            pred = model(X)
            sem_logits = pred[:, :SEM_CHANNELS]

            # semantic loss
            tgt_ids = Y_sem.clone()
            tgt_ids[tgt_ids == UNLABELED_ID] = 255
            loss_sem = sem_loss_fn(sem_logits, tgt_ids)

            def masked_weighted_huber(pred, target, mask, weight, delta=0.1, eps=1e-6):
                r = (pred - target)
                abs_r = r.abs()
                quad = torch.minimum(abs_r, torch.tensor(delta, device=pred.device))
                lin  = abs_r - quad
                huber = 0.5 * quad**2 / delta + lin
                huber = huber * mask * weight
                return huber.sum() / ((mask * weight).sum() + eps)

            pred_lin = pred[:, SEM_CHANNELS:SEM_CHANNELS+1].clamp_min(0)

            Y_elev_lin = Y_elev.clamp_min(0)

            loss_elev = masked_weighted_huber(pred_lin, Y_elev_lin, M, weight_map, delta=0.1)

            def grad_xy(img):
                gx = img[..., :, 1:] - img[..., :, :-1]
                gy = img[..., 1:, :] - img[..., :-1, :]
                return gx, gy

            gx, gy = grad_xy(pred_lin)
            with torch.no_grad():
                edges = (F.max_pool2d(Y_sem.float().unsqueeze(1), 3, 1, 1) != Y_sem.float().unsqueeze(1)).float()
                edges = F.interpolate(edges, size=pred_lin.shape[-2:], mode="nearest")
                smooth_w = (1.0 - 0.7 * edges).clamp_min(0.3)

            tv_loss = (smooth_w[..., :, 1:] * gx.abs()).mean() + (smooth_w[..., 1:, :] * gy.abs()).mean()
            loss = args.w_sem * loss_sem + args.w_elev * loss_elev + 0.02 * tv_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_steps   += 1
            epoch_loss_sum += float(loss.item())
            epoch_sem_sum  += float(loss_sem.item())
            epoch_elev_sum += float(loss_elev.item())
            epoch_tv_sum   += float(tv_loss.item())

            pbar.set_postfix(
                loss=f"{loss.item():.2f}",
                sem=f"{loss_sem.item():.2f}",
                elev=f"{loss_elev.item():.2f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}"
            )

        # ----- validation -----
        metrics = evaluate(model, val_loader, device, unlabeled_id=UNLABELED_ID)
        print(
            f"Epoch {epoch}/{args.epochs} [Validate] -> "
            f"mIoU: {metrics['miou']:.4f} | "
            f"MAE: {metrics['elev_mae']:.4f} | "
            f"RMSE: {metrics['elev_rmse']:.4f} | "
            f"BF Score: {metrics['bf_score']:.4f}"
        )

        with open(log_file_path, 'a') as f:
            f.write(
                f"{epoch},{metrics['miou']:.6f},{metrics['pix_acc']:.6f},"
                f"{metrics['elev_mae']:.6f},{metrics['elev_rmse']:.6f},"
                f"{metrics['bf_score']:.6f}\n"
            )

        lrs = [float(g["lr"]) for g in opt.param_groups]
        run_meta["lr_per_epoch"].append({
            "epoch": epoch,
            "lr_other": lrs[0],
            "lr_elev":  lrs[1] if len(lrs) > 1 else lrs[0],
        })
        lr_hist.append(lrs[0])

        run_meta["train_losses_per_epoch"].append({
            "epoch": epoch,
            "loss": epoch_loss_sum / max(1, epoch_steps),
            "sem":  epoch_sem_sum  / max(1, epoch_steps),
            "elev": epoch_elev_sum / max(1, epoch_steps),
            "tv":   epoch_tv_sum   / max(1, epoch_steps),
        })
        run_meta["val_metrics_per_epoch"].append({
            "epoch": epoch,
            "miou": float(metrics["miou"]),
            "pix_acc": float(metrics["pix_acc"]),
            "elev_mae": float(metrics["elev_mae"]),
            "elev_rmse": float(metrics["elev_rmse"]),
            "bf_score": float(metrics["bf_score"]),
        })

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),       
            "args": {**vars(args), "resolved_out": run_out},
            "sem_channels": SEM_CHANNELS,
            "include_mask": args.include_mask,
            "val_metrics": metrics,
            "best_miou": float(best_miou),             # <— helpful on resume
        }
        ckpt_path_epoch = os.path.join(run_out, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(ckpt, ckpt_path_epoch)

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            ckpt_path_best = os.path.join(run_out, "checkpoint_best.pt")
            torch.save(ckpt, ckpt_path_best)
            run_meta["checkpoints"]["best_by_miou"] = {
                "path": ckpt_path_best,
                "sha1": sha1_of_file(ckpt_path_best),
                "epoch": epoch,
                "miou": float(metrics["miou"]),
                "elev_mae": float(metrics["elev_mae"]),
                "elev_rmse": float(metrics["elev_rmse"]),
            }

        try:
            _save_training_plots(log_file_path, run_out, lr_hist=lr_hist)
        except Exception as e:
            print(f"[plot] warning: {e}")

        _safe_save_json(os.path.join(run_out, "manifest.training.json"), run_meta)

        sched.step()

    # ======== final test ========
    print("\nTraining finished. Evaluating best model on the test set...")

    ds_test = SeqFromWDS(shard_paths=test_shards, seq_len=args.seq_len,
                         shuffle_shards=False, include_mask=args.include_mask)
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, num_workers=max(1, args.workers//2), pin_memory=True,
        collate_fn=_collate, persistent_workers=(args.workers//2 > 0),
        prefetch_factor=(4 if args.workers//2 > 0 else None),
    )

    best_ckpt_path = os.path.join(run_out, "checkpoint_best.pt")
    if os.path.exists(best_ckpt_path):
        print(f"Loading best checkpoint from: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    else:
        print("Warning: 'checkpoint_best.pt' not found. Evaluating the last model state.")

    test_metrics = evaluate(model, test_loader, device, unlabeled_id=UNLABELED_ID)

    print("\n" + "="*40)
    print("          FINAL TEST SET RESULTS")
    print("="*40)
    print(f" >> mIoU:      {test_metrics['miou']:.4f}")
    print(f" >> Accuracy:  {test_metrics['pix_acc']:.4f}")
    print(f" >> Elev. MAE: {test_metrics['elev_mae']:.4f}")
    print(f" >> Elev. RMSE:{test_metrics['elev_rmse']:.4f}")
    print("="*40 + "\n")

    run_meta["test_metrics"] = {
        "miou": float(test_metrics["miou"]),
        "pix_acc": float(test_metrics["pix_acc"]),
        "elev_mae": float(test_metrics["elev_mae"]),
        "elev_rmse": float(test_metrics["elev_rmse"]),
    }
    run_meta["end_time_utc"] = datetime.utcnow().isoformat() + "Z"
    run_meta["elapsed_sec"]  = float(time.time() - run_start_ts)
    run_meta["files"] = {"training_log_csv": log_file_path, "plots_dir": run_out}
    _safe_save_json(os.path.join(run_out, "manifest.training.json"), run_meta)
    print(f"[manifest] wrote {os.path.join(run_out,'manifest.training.json')}")
    
    if test_shards: 
        print("\n--- Starting Inference on Test Set using Best Checkpoint ---") 
        inference_out_dir = os.path.join(run_out, "inference_on_test_set") 
        os.makedirs(inference_out_dir, exist_ok=True) 
        print(f"Saving inference results to: {inference_out_dir}") 
        try: 
            run_inference( checkpoint_path=best_ckpt_path, shards_list=test_shards, output_dir=inference_out_dir, device=str(device), seq_len=args.seq_len, model_name=args.model, sem_channels=SEM_CHANNELS, include_mask=args.include_mask ) 
            print(f"--- Inference complete. Results saved in: {inference_out_dir} ---") 
        except Exception as e: 
            print(f"\n[ERROR] An error occurred during the inference step: {e}") 
            import traceback 
            traceback.print_exc() 
    else: 
        print("\nSkipping inference step: No test shards were allocated.")
    
    print("Training complete.")

@torch.no_grad()
def evaluate(model, loader, device, unlabeled_id=0):
    model.eval()
    k = SEM_CHANNELS
    conf = torch.zeros((k, k), dtype=torch.int64, device=device)
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    valid_elev_count = 0.0
    valid_sem_count = 0

    boundary_tp, boundary_fp, boundary_fn = 0.0, 0.0, 0.0
    eps = 1e-6
    
    batches_processed = 0
    pbar_desc = "Evaluating"
    for X, (Y_sem, Y_elev), M in tqdm(loader, desc=pbar_desc, ncols=110, leave=False):
        batches_processed += 1
        X = X.to(device, non_blocking=True)
        Y_sem = Y_sem.to(device, non_blocking=True)
        Y_elev = Y_elev.to(device, non_blocking=True)
        M = M.to(device, non_blocking=True)

        pred = model(X)
        sem_logits = pred[:, :k]

        pred_ids = sem_logits.argmax(dim=1)
        valid_sem = (Y_sem != unlabeled_id)
        if valid_sem.any():
            t = Y_sem[valid_sem]
            p = pred_ids[valid_sem]
            valid_sem_count += t.numel() # Count valid pixels
            conf += torch.bincount(t * k + p, minlength=k*k).reshape(k, k)

        tp, fp, fn = calculate_batch_boundary_stats(pred_ids, Y_sem, unlabeled_id, tolerance=2)
        boundary_tp += tp.item()
        boundary_fp += fp.item()
        boundary_fn += fn.item()
        
        pred_elev_real = pred[:, k:k+1].clamp(min=0)

        diff = (pred_elev_real - Y_elev) * M
        abs_err_sum += diff.abs().sum().item()
        sq_err_sum  += (diff ** 2).sum().item()
        valid_elev_count += M.sum().item()

    if batches_processed == 0:
        print("  -> WARNING: Validation DataLoader was empty. No data was evaluated.")
        return {"miou": 0.0, "pix_acc": 0.0, "elev_mae": 0.0, "elev_rmse": 0.0}

    # semantics
    diag = conf.diag().float()
    tp_fp = conf.sum(dim=0).float()
    tp_fn = conf.sum(dim=1).float()
    denom = (tp_fp + tp_fn - diag).clamp_min(1)
    iou = diag / denom

    cls = torch.arange(k, device=device) != unlabeled_id
    miou = iou[cls].mean().item() if cls.any() else 0.0 # Handle case with no valid classes
    pix_acc = (diag.sum() / conf.sum().clamp_min(1)).item()

    precision = boundary_tp / (boundary_tp + boundary_fp + eps)
    recall = boundary_tp / (boundary_tp + boundary_fn + eps)
    bf_score = 2 * (precision * recall) / (precision + recall + eps)
    
    # elevation
    mae = abs_err_sum / max(valid_elev_count, 1.0)
    rmse = (sq_err_sum / max(valid_elev_count, 1.0)) ** 0.5

    return {
        "miou": miou, "pix_acc": pix_acc,
        "elev_mae": mae, "elev_rmse": rmse,
        "bf_score": bf_score 
    }
@torch.no_grad()
def calculate_batch_boundary_stats(pred_ids, gt_ids, unlabeled_id, tolerance=2):
    device = pred_ids.device
    kernel = torch.ones(1, 1, 3, 3, device=device)
    valid_mask = (gt_ids != unlabeled_id)
    
    gt_boundaries = (F.max_pool2d(gt_ids.float().unsqueeze(1), 3, 1, 1) != gt_ids.float().unsqueeze(1)).squeeze(1)
    gt_boundaries &= valid_mask # Only consider boundaries in valid areas

    pred_boundaries = (F.max_pool2d(pred_ids.float().unsqueeze(1), 3, 1, 1) != pred_ids.float().unsqueeze(1)).squeeze(1)
    
    if tolerance > 0:
        k_size = 2 * tolerance + 1
        gt_boundaries_dilated = F.max_pool2d(gt_boundaries.float().unsqueeze(1), k_size, 1, padding=tolerance).squeeze(1) > 0
    else:
        gt_boundaries_dilated = gt_boundaries
        
    tp = (pred_boundaries & gt_boundaries_dilated).sum()
    fp = (pred_boundaries & ~gt_boundaries_dilated).sum()
    fn = (gt_boundaries & ~pred_boundaries).sum()
    
    return tp, fp, fn

def build_argparser():
    p = argparse.ArgumentParser(description="Train ConvLSTM U-Net with 14ch one-hot semantics (+elev).")
    p.add_argument("--shards", required=True, help="Directory with .tar shards or a pattern")
    p.add_argument("--out", default="/media/slsecret/T7/carla3/runs/town0", help="Output directory")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4)

    p.add_argument("--base", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--include-mask", type=lambda s: s.lower() in ("1","true","yes","y"), default=True,
                   help="Append training-valid mask channel to input (default: true).")
    p.add_argument("--w-elev", type=float, default=1.0, help="Weight for Elevation L1.")
    p.add_argument("--w-sem", type=float, default=1.0, help="Weight for Semantics CE.")
    p.add_argument("--elev-l1-weight", type=float, default=0.5,
                help="Weight for the additional L1 term when using huber_l1.")
    p.add_argument("--split-seed", type=int, default=357,
                   help="Seed controlling deterministic session split.")
    
    p.add_argument("--loss-type", type=str, default="focal", choices=["ce", "focal"],
                   help="Type of semantic loss to use: 'ce' or 'focal'.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focusing parameter gamma for Focal Loss.")
    p.add_argument("--auto-class-weights", type=lambda s: s.lower() in ("1","true","yes","y"), default=True,
                   help="Automatically calculate class weights for Focal Loss from training data (default: true).")
    p.add_argument(
        "--model",
        default="unet",
        choices=["cnn", "unet", "unet_attn", "deeplabv3p"],
        help="Backbone to use: cnn (two-stage correction), unet (ConvLSTM UNet), "
             "unet_attn (Attention ConvLSTM UNet), deeplabv3p (DeepLabV3+)."
    )
    p.add_argument(
        "--resume",
        default="",
        help="Path to a checkpoint .pt or a directory containing checkpoints. If a directory, the latest checkpoint_epoch_*.pt will be used."
    )
    p.add_argument(
        "--allow-new-epochs",
        type=lambda s: s.lower() in ("1","true","yes","y"),
        default=True,
        help="If true and --epochs <= last finished epoch, run only eval instead of training."
    )
    return p

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("--- Multiprocessing start method set to 'spawn' ---")
    except RuntimeError:
        pass

    args = build_argparser().parse_args()
    train(args)

"""
python3 train2.py \
--shards /media/slsecret/T7/carla3/data_split357 \
--workers 4 \
--include-mask true \
--w-sem 1.0 --w-elev 1.0 --out /media/slsecret/T7/carla3/runs/all357 --epochs 1

--resume /media/slsecret/T7/carla3/runs/all357_unet
"""