#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference with the ConvLSTM U-Net trained by train.py.

Inputs:
  - --shards: directory containing .tar shards OR a glob/brace pattern
              (same format as training). Each sample must include:
                tr_rgb.png  (uint8 [H,W,3])
                tr_elev.npy (float32 [H,W])
              Optional files (ignored by inference): valid.png, gt_*.*
  - --checkpoint: path to a training checkpoint saved by train.py

Outputs:
  - New WebDataset shards written to --out, each sample with:
      pred_rgb.png   (uint8 [H,W,3], RGB)
      pred_elev.npy  (float32 [H,W])
      meta.json      (copied from input if present)
    Keys are preserved (one output sample per input time step whenever a full
    T-frame context is available). The first (T-1) samples per session are skipped
    because the temporal context is insufficient.

Example:
  python3 infer.py \
      --shards /media/slsecret/T7/carla3/data/town1/gridmap_wds2/ \
      --checkpoint runs/gridmap_convLSTM/checkpoint_epoch_020.pt \
      --out /media/slsecret/T7/carla3/runs/infer_town1 \
      --seq-len 4 --device cuda --maxcount 5000
"""

import os
import io
import json
import argparse
from collections import defaultdict, deque

import numpy as np
import cv2
import torch
import torch.nn as nn

import webdataset as wds
from tqdm import tqdm

# Import model & helpers from training script
from train_rgb import UNetConvLSTM, resolve_shards, decode_png_rgb, decode_npy, parse_key


def _rgb_to_png_bytes(rgb_float01: np.ndarray) -> bytes:
    """[H,W,3] float32 in [0,1] RGB -> PNG bytes in BGR (for OpenCV encoder)."""
    x = np.clip(rgb_float01 * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return buf.tobytes()


def _npy_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize numpy array to .npy bytes."""
    with io.BytesIO() as f:
        np.save(f, arr)
        return f.getvalue()


def _sanitize_elev(elev: np.ndarray):
    """
    Replace NaN/Inf with 0.0 and return (elev_clean, valid_mask_float32).
    valid_mask is 1.0 where finite, 0.0 otherwise.
    """
    valid = np.isfinite(elev)
    clean = np.nan_to_num(elev, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return clean, valid.astype(np.float32)


def _open_wds(shards_pattern: str):
    """Open a WebDataset stream without shuffling; be resilient across versions."""
    shard_paths = resolve_shards(shards_pattern)
    dataset = wds.WebDataset(
        shard_paths,
        shardshuffle=False,
        resampled=False,
        empty_check=False,
    )
    # Split by node/worker if available, but keep order stable within a node/worker
    if hasattr(wds, "split_by_node"):
        dataset = wds.split_by_node(dataset)
    if hasattr(wds, "split_by_worker"):
        dataset = wds.split_by_worker(dataset)
    if hasattr(dataset, "with_length"):
        dataset = dataset.with_length(None)
    return dataset


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    # Robustly read base channels from the checkpoint args
    base = 32
    try:
        base = int(ckpt.get("args", {}).get("base", 32))
    except Exception:
        pass
    model = UNetConvLSTM(in_ch=5, base=base, out_ch=4).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer(args):
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # Model
    model = load_model(args.checkpoint, device)

    # IO
    dataset = _open_wds(args.shards)
    writer_pattern = os.path.join(args.out, "shard-%06d.tar")
    # Create writer (WebDataset version compatible)
    if hasattr(wds, "ShardWriter"):
        # Newer API: supports automatic shard rotation via maxcount
        sink = wds.ShardWriter(writer_pattern, maxcount=args.maxcount)
    else:
        # Older API: TarWriter may not accept maxcount; write a single tar
        single_tar = os.path.join(args.out, "predictions.tar")
        sink = wds.TarWriter(single_tar)

    # Per-session temporal buffers
    hist = defaultdict(lambda: deque(maxlen=args.seq_len))

    processed = 0
    with torch.inference_mode():
        for sample in tqdm(dataset, desc="Infer", ncols=100):
            key = sample.get("__key__")
            if key is None:
                continue
            sess, idx = parse_key(key)

            # Decode inputs
            tr_rgb = decode_png_rgb(sample["tr_rgb.png"]).astype(np.float32) / 255.0  # [H,W,3]
            tr_elev = decode_npy(sample["tr_elev.npy"]).astype(np.float32)            # [H,W]
            tr_elev, tr_mask = _sanitize_elev(tr_elev)                                  # [H,W], [H,W]

            meta = None
            if "meta.json" in sample:
                try:
                    meta = json.loads(sample["meta.json"].decode("utf-8"))
                except Exception:
                    meta = None

            # Push to history for this session
            hist[sess].append({
                "tr_rgb": tr_rgb,
                "tr_elev": tr_elev,
                "tr_mask": tr_mask,
                "meta": meta,
                "key": key,
            })

            if len(hist[sess]) < args.seq_len:
                continue  # not enough context yet

            # Build input tensor from the last T frames
            frames = list(hist[sess])[-args.seq_len:]
            H, W, _ = frames[-1]["tr_rgb"].shape

            X_np = []
            for fr in frames:
                x = np.concatenate([
                    fr["tr_rgb"],
                    fr["tr_elev"][..., None],
                    fr["tr_mask"][..., None],
                ], axis=-1)  # [H,W,5]
                X_np.append(x)
            X_np = np.stack(X_np, axis=0).astype(np.float32)  # [T,H,W,5]
            X_t = torch.from_numpy(np.transpose(X_np, (0, 3, 1, 2))).unsqueeze(0).to(device)  # [1,T,5,H,W]

            # Forward
            preds = model(X_t)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            P_rgb = preds[:, :3].clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()   # [H,W,3]
            P_elev = preds[:, 3:4].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)  # [H,W]

            # Prepare output sample
            out = {"__key__": frames[-1]["key"]}
            out["pred_rgb.png"] = _rgb_to_png_bytes(P_rgb)
            out["pred_elev.npy"] = _npy_to_bytes(P_elev)
            if frames[-1]["meta"] is not None:
                out["meta.json"] = json.dumps(frames[-1]["meta"], separators=(",", ":")).encode("utf-8")

            sink.write(out)
            processed += 1

    sink.close()
    print(f"Done. Wrote {processed} predictions to: {args.out}")


def build_argparser():
    p = argparse.ArgumentParser(description="ConvLSTM U-Net inference over WebDataset shards")
    p.add_argument("--shards", required=True,
                   help="Directory of .tar shards or a glob/brace pattern (same as training)")
    p.add_argument("--checkpoint", required=True, help="Path to a training checkpoint .pt from train.py")
    p.add_argument("--out", required=True, help="Output directory where new shards will be written")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    p.add_argument("--seq-len", type=int, default=4, help="Temporal context T (must match training)")
    p.add_argument("--maxcount", type=int, default=10000, help="Max samples per output shard before rotation")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    infer(args)
