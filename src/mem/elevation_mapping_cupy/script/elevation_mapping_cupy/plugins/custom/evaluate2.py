#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference (14ch semantics):
- Input per frame: [onehot14, elev(1), mask(1?)] stacked over T frames
- Output per frame: sem_logits14 + elev(1)
- We postprocess to:
    * pred_onehot14.npy  (H,W,14, uint8 {0,1}, from argmax over channels)
    * pred_color14.png   (H,W,3, uint8) via remap_color.onehot14_to_color
    * pred_elev.npy      (H,W, float32)
    * (optional) pred_sem_logits.npy (H,W,14, float32) if --save-logits true

Usage:
  python3 evaluate2.py \
    --shards /media/slsecret/T7/carla3/data/town0/gridmap_wds_remapped \
    --checkpoint /media/slsecret/T7/carla3/runs/town0_unet_attention/checkpoint_epoch_010.pt \
    --out /media/slsecret/T7/carla3/runs/infer_town0_unet_attention \
    --seq-len 4 --device cuda \
    --sem-channels 14 --include-mask true \
    --save-logits false --maxcount 10000
"""

import os
import io
import json
import argparse
from collections import defaultdict, deque

import warnings, numpy as np
warnings.filterwarnings("ignore", message="The value of the smallest subnormal", category=UserWarning)
import cv2
import torch
import torch.nn as nn
import webdataset as wds
from tqdm import tqdm

try:
    from .unet_conv_lstm import UNetConvLSTM
    from .unet_attention import AttentionUNetConvLSTM
    from .deeplabv3plus import DeepLabV3Plus
    from .cnn import CNNCorrectionNetSingle
    from .shard_utils import resolve_shards, decode_npy, parse_key
    from .color_mapping import onehot14_to_color
except ImportError:
    from unet_conv_lstm import UNetConvLSTM
    from unet_attention import AttentionUNetConvLSTM
    from deeplabv3plus import DeepLabV3Plus
    from cnn import CNNCorrectionNetSingle
    from shard_utils import resolve_shards, decode_npy, parse_key
    from color_mapping import onehot14_to_color

def _model_from_name(name: str, C_in: int, C_out: int, base: int):
    if name == "unet":
        return UNetConvLSTM(in_ch=C_in, base=base, out_ch=C_out)
    if name == "unet_attn":
        return AttentionUNetConvLSTM(in_ch=C_in, base=base, out_ch=C_out)
    if name == "deeplabv3p":
        return DeepLabV3Plus(in_ch=C_in, out_ch=C_out)
    if name == "cnn":
        return CNNCorrectionNetSingle(
            in_ch_per_frame=C_in, base=base,
            blocks_stage1=4, blocks_stage2=6,
            aspp_rates=(1,2,4), use_identity_correction=True, return_edit=False
        )
    raise ValueError(f"Unknown model name: {name}")

def _build_model_from_ckpt(ckpt_path: str,
                           device: torch.device,
                           sem_channels_cli: int,
                           include_mask_cli: bool,
                           model_cli: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    args_in = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    base = int(args_in.get("base", 32))
    trained_sem_channels = int(ckpt.get("sem_channels", sem_channels_cli))
    trained_include_mask = bool(ckpt.get("include_mask", include_mask_cli))

    if model_cli == "auto":
        model_name = args_in.get("model", "unet_attn")
    else:
        model_name = model_cli

    C_in  = trained_sem_channels + 1 + (1 if trained_include_mask else 0)
    C_out = trained_sem_channels + 1

    model = _model_from_name(model_name, C_in=C_in, C_out=C_out, base=base).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[INFO] Loaded checkpoint '{ckpt_path}' with:")
    print(f"       model={model_name}, base={base}, sem_channels={trained_sem_channels}, include_mask={trained_include_mask}")
    return model, trained_sem_channels, trained_include_mask

def _elev_to_heatmap_png(elev: np.ndarray, compression: int = 3,
                         vmin: float = None, vmax: float = None) -> bytes:
    """
    Convert float32 elevation (H,W) to a heatmap PNG using OpenCV JET.
    NaN/inf -> black. If vmin/vmax are None, use per-frame min/max of valid pixels.
    """
    e = np.asarray(elev, dtype=np.float32)
    mask = np.isfinite(e)

    if not np.any(mask):
        img_bgr = np.zeros((*e.shape, 3), dtype=np.uint8)
    else:
        lo = float(np.min(e[mask])) if vmin is None else float(vmin)
        hi = float(np.max(e[mask])) if vmax is None else float(vmax)
        rng = max(hi - lo, 1e-6)

        norm = np.zeros_like(e, dtype=np.float32)
        norm[mask] = (e[mask] - lo) / rng
        img_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img_bgr[~mask] = 0

    ok, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise ValueError("Failed to encode elevation heatmap PNG")
    return buf.tobytes()

def _rgb_to_png_bytes(rgb_uint8_hwc: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb_uint8_hwc, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("Failed to encode PNG")
    return buf.tobytes()

def _npy_to_bytes(arr: np.ndarray) -> bytes:
    with io.BytesIO() as f:
        np.save(f, arr, allow_pickle=False)
        return f.getvalue()

def _sanitize_elev(elev: np.ndarray):
    valid = np.isfinite(elev)
    clean = np.nan_to_num(elev, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return clean, valid.astype(np.float32)

def _open_wds(shards_pattern: str, recursive: bool, only_remapped: bool):
    shard_paths = resolve_shards(shards_pattern, recursive=recursive, only_remapped=only_remapped)
    dataset = wds.WebDataset(shard_paths, shardshuffle=False, resampled=False, empty_check=False)
    if hasattr(wds, "split_by_node"):
        dataset = wds.split_by_node(dataset)
    if hasattr(wds, "split_by_worker"):
        dataset = wds.split_by_worker(dataset)
    if hasattr(dataset, "with_length"):
        dataset = dataset.with_length(None)
    return dataset

def infer(args):
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    os.makedirs(args.out, exist_ok=True)

    model, semC, use_mask = _build_model_from_ckpt(
        args.checkpoint, device, args.sem_channels, args.include_mask, args.model
    )

    dataset = _open_wds(args.shards, recursive=args.recursive, only_remapped=args.only_remapped)
    writer_pattern = os.path.join(args.out, "shard-%06d.tar")
    sink = wds.ShardWriter(writer_pattern, maxcount=args.maxcount)

    hist = defaultdict(lambda: deque(maxlen=args.seq_len))

    processed = 0
    with torch.inference_mode():
        for sample in tqdm(dataset, desc="Infer14", ncols=100):
            key = sample.get("__key__")
            if key is None:
                continue
            sess, idx = parse_key(key)

            if "tr_onehot14.npy" not in sample:
                raise KeyError("Input sample missing 'tr_onehot14.npy'.")
            tr_onehot = decode_npy(sample["tr_onehot14.npy"]).astype(np.float32)
            if tr_onehot.ndim != 3 or tr_onehot.shape[2] != semC:
                raise ValueError(f"tr_onehot14.npy has shape {tr_onehot.shape}, expected (H,W,{semC})")
            
            tr_elev = decode_npy(sample["tr_elev.npy"]).astype(np.float32)
            tr_elev_sanitized, tr_mask = _sanitize_elev(tr_elev)

            if "gt_rgb.png" not in sample:
                print(f"Warning: 'gt_rgb.png' not found for key {key}.")
                gt_rgb_bytes = None
            else:
                gt_rgb_bytes = sample["gt_rgb.png"]
            
            if "gt_elev.npy" not in sample:
                print(f"Warning: 'gt_elev.npy' not found for key {key}.")
                gt_elev_data = None
            else:
                gt_elev_data = decode_npy(sample["gt_elev.npy"]).astype(np.float32)

            meta = None
            if "meta.json" in sample:
                try: meta = json.loads(sample["meta.json"].decode("utf-8"))
                except Exception: meta = None

            hist[sess].append({
                "tr_sem": tr_onehot,
                "tr_elev": tr_elev_sanitized,
                "tr_mask": tr_mask if use_mask else None,
                "gt_rgb_png": gt_rgb_bytes,
                "gt_elev": gt_elev_data, # MODIFICATION: Store GT elevation
                "meta": meta,
                "key": key
            })

            if len(hist[sess]) < args.seq_len:
                continue

            frames = list(hist[sess])[-args.seq_len:]
            H, W, _ = frames[-1]["tr_sem"].shape

            X_np = []
            for fr in frames:
                parts = [fr["tr_sem"], fr["tr_elev"][..., None]]
                if use_mask:
                    parts.append(fr["tr_mask"][..., None])
                x = np.concatenate(parts, axis=-1)
                X_np.append(x)
            X_np = np.stack(X_np, axis=0).astype(np.float32)
            X_t  = torch.from_numpy(np.transpose(X_np, (0,3,1,2))).unsqueeze(0).to(device)

            # Forward
            preds = model(X_t)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            sem_logits = preds[:, :semC].squeeze(0).permute(1,2,0).cpu().numpy()
            pred_elev_log = preds[:, semC:semC+1]
            pred_elev_real = torch.expm1(pred_elev_log).clamp(min=0)
            pred_elev = pred_elev_real.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            
            pred_elev_viz_png = _elev_to_heatmap_png(pred_elev, compression=3)
            argmax_idx = np.argmax(sem_logits, axis=2)
            pred_onehot = np.eye(semC, dtype=np.uint8)[argmax_idx]
            pred_color = onehot14_to_color(pred_onehot.astype(np.float32))
            
            current_gt_frame = frames[-1]

            gt_color_png = current_gt_frame["gt_rgb_png"]

            gt_elev = current_gt_frame["gt_elev"]
            gt_elev_viz_png = None
            if gt_elev is not None:
                gt_elev_viz_png = _elev_to_heatmap_png(gt_elev, compression=3)

            out = {"__key__": frames[-1]["key"]}
            
            if gt_color_png is not None:
                out["gt_color14.png"] = gt_color_png
            if gt_elev_viz_png is not None:
                out["gt_elev_viz.png"] = gt_elev_viz_png
            
            out["pred_color14.png"] = _rgb_to_png_bytes(pred_color)
            out["pred_elev_viz.png"] = pred_elev_viz_png

            sink.write(out)
            processed += 1

    sink.close()
    print(f"Done. Wrote {processed} predictions to: {args.out}")

def build_argparser():
    p = argparse.ArgumentParser(description="ConvLSTM U-Net inference (14ch semantics) over WebDataset shards")
    p.add_argument("--shards", required=True, help="Directory of .tar shards or a glob/brace pattern")
    p.add_argument("--checkpoint", required=True, help="Path to a training checkpoint (.pt) from train.py")
    p.add_argument("--out", required=True, help="Output directory for new shards")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seq-len", type=int, default=4, help="Temporal context T (must match training)")
    p.add_argument("--maxcount", type=int, default=10000, help="Max samples per output shard before rotation")
    p.add_argument("--sem-channels", type=int, default=14, help="Semantic channels (default 14)")
    p.add_argument("--include-mask", type=lambda s: s.lower() in ("1","true","yes","y"),
                   default=True, help="Whether mask channel was used during training")
    p.add_argument("--save-logits", type=lambda s: s.lower() in ("1","true","yes","y"),
                   default=False, help="Also save raw sem logits as pred_sem_logits.npy")
    p.add_argument(
        "--model",
        default="auto",
        choices=["auto", "unet", "unet_attn", "deeplabv3p", "cnn"],
        help="Backbone for inference. 'auto' = use model saved in checkpoint if available, "
             "otherwise defaults to unet_attn."
    )
    p.add_argument("--recursive", type=lambda s: s.lower() in ("1","true","yes","y"),
                   default=True, help="Recurse into subdirectories when --shards is a directory.")
    p.add_argument("--only-remapped", type=lambda s: s.lower() in ("1","true","yes","y"),
                   default=True, help="Only use shards whose directory path contains 'remapped'.")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    infer(args)