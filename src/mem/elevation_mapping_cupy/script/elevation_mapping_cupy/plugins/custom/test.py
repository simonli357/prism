#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, time
from datetime import datetime
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .evaluate3 import run_inference
    from .unet_conv_lstm import UNetConvLSTM
    from .deeplabv3plus import DeepLabV3Plus
    from .color_mapping import SEM_CHANNELS
    from .shard_utils import SeqFromWDS, resolve_shards
except ImportError:
    from evaluate3 import run_inference
    from unet_conv_lstm import UNetConvLSTM
    from deeplabv3plus import DeepLabV3Plus
    from color_mapping import SEM_CHANNELS
    from shard_utils import SeqFromWDS, resolve_shards


def _collate(batch):
    Xs, Y_sems, Y_elevs, Ms = [], [], [], []
    for X, (Y_sem, Y_elev), M in batch:
        # X: [T,H,W,C] -> [T,C,H,W]
        Xs.append(torch.from_numpy(np.transpose(X, (0,3,1,2))).float())
        # semantics: [H,W] long
        Y_sems.append(torch.from_numpy(Y_sem).long())
        # elevation: [H,W,1] -> [1,H,W]
        Y_elevs.append(torch.from_numpy(np.transpose(Y_elev, (2,0,1))).float())
        # mask: [H,W] -> [1,H,W]
        Ms.append(torch.from_numpy(M[None, ...]).float())
    return torch.stack(Xs), (torch.stack(Y_sems), torch.stack(Y_elevs)), torch.stack(Ms)


@torch.no_grad()
def calculate_batch_boundary_stats(pred_ids, gt_ids, unlabeled_id, tolerance=2):
    device = pred_ids.device
    valid_mask = (gt_ids != unlabeled_id)

    gt_boundaries = (F.max_pool2d(gt_ids.float().unsqueeze(1), 3, 1, 1) != gt_ids.float().unsqueeze(1)).squeeze(1)
    gt_boundaries &= valid_mask

    pred_boundaries = (F.max_pool2d(pred_ids.float().unsqueeze(1), 3, 1, 1) != pred_ids.float().unsqueeze(1)).squeeze(1)

    if tolerance > 0:
        k_size = 2 * tolerance + 1
        gt_boundaries_dilated = F.max_pool2d(gt_boundaries.float().unsqueeze(1), k_size, 1,
                                             padding=tolerance).squeeze(1) > 0
    else:
        gt_boundaries_dilated = gt_boundaries

    tp = (pred_boundaries & gt_boundaries_dilated).sum()
    fp = (pred_boundaries & ~gt_boundaries_dilated).sum()
    fn = (gt_boundaries & ~pred_boundaries).sum()
    return tp, fp, fn


@torch.no_grad()
def evaluate(model, loader, device, elev_output_space="log", unlabeled_id=0):
    """
    elev_output_space: "log" (model outputs log(1+elev)) or "linear" (model outputs elev in meters)
    """
    model.eval()
    k = SEM_CHANNELS

    conf = torch.zeros((k, k), dtype=torch.int64, device=device)
    abs_err_sum = 0.0
    sq_err_sum  = 0.0
    valid_elev_count = 0.0

    boundary_tp = boundary_fp = boundary_fn = 0.0
    eps = 1e-6

    batches = 0
    for X, (Y_sem, Y_elev), M in tqdm(loader, desc="Evaluating (test)", ncols=110):
        batches += 1
        X      = X.to(device, non_blocking=True)          # [B,T,C,H,W]
        Y_sem  = Y_sem.to(device, non_blocking=True)      # [B,H,W]
        Y_elev = Y_elev.to(device, non_blocking=True)     # [B,1,H,W]
        M      = M.to(device, non_blocking=True)          # [B,1,H,W]

        pred = model(X)                                   # [B, K+1, H, W]
        sem_logits = pred[:, :k]                          # [B,K,H,W]
        elev_head  = pred[:, k:k+1]                       # [B,1,H,W]

        # semantics
        pred_ids = sem_logits.argmax(dim=1)               # [B,H,W]
        valid_sem = (Y_sem != unlabeled_id)
        if valid_sem.any():
            t = Y_sem[valid_sem]
            p = pred_ids[valid_sem]
            conf += torch.bincount(t * k + p, minlength=k*k).reshape(k, k)

        tp, fp, fn = calculate_batch_boundary_stats(pred_ids, Y_sem, unlabeled_id, tolerance=2)
        boundary_tp += tp.item(); boundary_fp += fp.item(); boundary_fn += fn.item()

        # elevation: choose linearization based on head space
        if elev_output_space == "log":
            pred_elev_real = torch.expm1(elev_head).clamp(min=0)
        else:
            pred_elev_real = elev_head.clamp(min=0)

        diff = (pred_elev_real - Y_elev) * M
        abs_err_sum += diff.abs().sum().item()
        sq_err_sum  += (diff ** 2).sum().item()
        valid_elev_count += M.sum().item()

    if batches == 0:
        return {"miou": 0.0, "pix_acc": 0.0, "elev_mae": 0.0, "elev_rmse": 0.0, "bf_score": 0.0}

    # semantics metrics
    diag  = conf.diag().float()
    tp_fp = conf.sum(dim=0).float()
    tp_fn = conf.sum(dim=1).float()
    denom = (tp_fp + tp_fn - diag).clamp_min(1)
    iou   = diag / denom

    cls_mask = torch.arange(k, device=device) != unlabeled_id
    miou = iou[cls_mask].mean().item() if cls_mask.any() else 0.0
    pix_acc = (diag.sum() / conf.sum().clamp_min(1)).item()

    precision = boundary_tp / (boundary_tp + boundary_fp + eps)
    recall    = boundary_tp / (boundary_tp + boundary_fn + eps)
    bf_score  = 2 * (precision * recall) / (precision + recall + eps)

    # elevation metrics
    mae  = abs_err_sum / max(valid_elev_count, 1.0)
    rmse = (sq_err_sum  / max(valid_elev_count, 1.0)) ** 0.5

    return {"miou": miou, "pix_acc": pix_acc, "elev_mae": mae, "elev_rmse": rmse, "bf_score": bf_score}


def build_model_from_ckpt(ckpt, device, C_in, C_out):
    args = ckpt.get("args", {})
    model_name = args.get("model", "unet")
    base = args.get("base", 64)

    if model_name == "unet":
        model = UNetConvLSTM(in_ch=C_in, base=base, out_ch=C_out)
    elif model_name == "deeplabv3p":
        model = DeepLabV3Plus(in_ch=C_in, out_ch=C_out)
    else:
        # Fall back to UNet for unknown strings
        print(f"[WARN] Unknown model '{model_name}', defaulting to UNetConvLSTM.")
        model = UNetConvLSTM(in_ch=C_in, base=base, out_ch=C_out)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, model_name, base


def main():
    p = argparse.ArgumentParser(description="Evaluate best checkpoint on the TEST split.")
    p.add_argument("--run-dir", required=True,
                   help="Directory of a training run that contains checkpoint_best.pt and manifests.")
    p.add_argument("--shards", required=True,
                   help="Root directory that contains 'test' shards subfolder (same as training).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=None,
                   help="Override seq_len (otherwise taken from checkpoint args).")
    p.add_argument("--include-mask", type=str, default=None,
                   help="Override include_mask true/false (otherwise from checkpoint args).")
    p.add_argument("--elev-output", type=str, default="log", choices=["log", "linear"],
                   help="Space of elevation head output: 'log' (=exp(m)-1 at eval) or 'linear'.")
    p.add_argument("--save-json", type=str, default="",
                   help="Optional path to write metrics JSON (default: none).")
    p.add_argument("--pred-out", type=str, default="/media/slsecret/T7/carla3/runs/all357/inference_on_test_set",
                   help="If set, write test predictions (WebDataset shards) to this output directory.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ckpt_path = os.path.join(args.run_dir, "checkpoint_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    print(f"Loading best checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}

    # Resolve seq_len / include_mask from ckpt unless overridden
    seq_len = args.seq_len if args.seq_len is not None else ckpt_args.get("seq_len", 4)
    include_mask = (
        (args.include_mask.lower() in ("1","true","yes","y")) if isinstance(args.include_mask, str)
        else ckpt_args.get("include_mask", True)
    )
    print(f"Device: {device} | seq_len={seq_len} | include_mask={include_mask}")

    # IO channels
    C_in  = SEM_CHANNELS + 1 + (1 if include_mask else 0)
    C_out = SEM_CHANNELS + 1

    # Build model
    model, model_name, base = build_model_from_ckpt(ckpt, device, C_in, C_out)
    print(f"[INFO] Loaded checkpoint with: model={model_name}, base={base}, sem_channels={SEM_CHANNELS}, include_mask={include_mask}")

    # Dataset: TEST split under shards root
    test_dir = os.path.join(args.shards, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"No 'test' directory under shards root: {test_dir}")

    test_shards = resolve_shards(test_dir, recursive=False, only_remapped=False)
    if not test_shards:
        raise FileNotFoundError(f"No shard files found under: {test_dir}")
    ds_test = SeqFromWDS(
        shard_paths=test_shards,
        seq_len=seq_len,
        shuffle_shards=False,
        include_mask=include_mask
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        num_workers=max(1, args.workers),
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=(args.workers > 0),
        prefetch_factor=(4 if args.workers > 0 else None),
    )

    # If the ckpt recorded the training setting, prefer it (unless user overrides)
    elev_output_space = args.elev_output
    if isinstance(ckpt_args, dict):
        # accept either explicit flag stored later or infer from loss type
        if ckpt_args.get("elev_output_space") in ("log", "linear"):
            elev_output_space = ckpt_args["elev_output_space"]
        elif ckpt_args.get("elev_loss") == "log_huber":
            elev_output_space = "log"
        elif ckpt_args.get("elev_loss") in ("huber", "huber_l1"):
            # In your mixed mode you still used a log head; keep 'log' unless you fully switched heads.
            elev_output_space = elev_output_space  # leave as CLI default
    print(f"[INFO] Evaluating with elev_output_space='{elev_output_space}'")

    # ---- Run evaluation
    t0 = time.time()
    metrics = evaluate(model, test_loader, device, elev_output_space=elev_output_space, unlabeled_id=0)
    dt = time.time() - t0

    print("\n" + "="*40)
    print("        TEST SET RESULTS (best ckpt)")
    print("="*40)
    print(f" >> mIoU:      {metrics['miou']:.4f}")
    print(f" >> Accuracy:  {metrics['pix_acc']:.4f}")
    print(f" >> Elev. MAE: {metrics['elev_mae']:.4f}")
    print(f" >> Elev. RMSE:{metrics['elev_rmse']:.4f}")
    print(f" >> BF Score:  {metrics['bf_score']:.4f}")
    print("="*40)
    print(f"[done] Evaluated in {dt/60:.1f} min")

    if args.save_json:
        out = {
            "type": "test_only",
            "run_dir": args.run_dir,
            "ckpt_path": ckpt_path,
            "time_utc": datetime.utcnow().isoformat() + "Z",
            "device": str(device),
            "seq_len": int(seq_len),
            "include_mask": bool(include_mask),
            "elev_output_space": elev_output_space,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[metrics] wrote JSON to {args.save_json}")

    if args.pred_out:
      os.makedirs(args.pred_out, exist_ok=True)
      print(f"\n--- Writing test predictions to: {args.pred_out} ---")
      try:
          run_inference(
              checkpoint_path=ckpt_path,
              shards_list=test_shards,
              output_dir=args.pred_out,
              device=str(device),
              seq_len=seq_len,
              model_name=model_name,
              sem_channels=SEM_CHANNELS,
              include_mask=include_mask,
          )
          print(f"--- Inference complete. Results saved in: {args.pred_out} ---")
      except Exception as e:
          import traceback
          print(f"[ERROR] An error occurred while writing predictions: {e}")
          traceback.print_exc()

if __name__ == "__main__":
    main()

"""
python3 test.py \
--run-dir /media/slsecret/T7/carla3/runs/all357_unet \
--shards   /media/slsecret/T7/carla3/data_split357 \
--device cuda \
--batch-size 8 \
--workers 4 \
--elev-output linear \
--save-json /media/slsecret/T7/carla3/runs/all357_unet/test_metrics.json \
--pred-out  /media/slsecret/T7/carla3/runs/all357_unet/inference_on_test_set2
"""