#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a ConvLSTM U-Net to translate Training grid-maps to Ground-Truth.

Data format (WebDataset shards):
  __key__         : e.g. run1_000000123
  gt_rgb.png      : uint8 [H,W,3] RGB (PNG)
  tr_rgb.png      : uint8 [H,W,3] RGB (PNG)
  gt_elev.npy     : float32 [H,W]
  tr_elev.npy     : float32 [H,W]
  valid.png       : (optional) uint8 [H,W] 0/255 mask (GT-valid)
  meta.json       : metadata

Model:
  - Input:  last T frames of training maps, per-frame channels [R,G,B,elev, M]
  - Output: 4 channels for the current GT frame [R,G,B,elev]
  - Loss:   L1 on RGB (masked), L1 on Elevation (masked)
"""

import os
import glob
import io
import json
import math
import argparse
from collections import defaultdict, deque

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import webdataset as wds
from tqdm import tqdm

def resolve_shards(path_or_pattern: str):
    if os.path.isdir(path_or_pattern):
        shards = sorted(glob.glob(os.path.join(path_or_pattern, "*.tar")))
        if not shards:
            raise FileNotFoundError(f"No .tar shards found in directory: {path_or_pattern}")
        print(f"[INFO] Found {len(shards)} shards in {path_or_pattern}")
        return shards
    else:
        # Assume it’s a glob or brace pattern
        shards = sorted(glob.glob(path_or_pattern))
        if not shards:
            raise FileNotFoundError(f"No shards found for pattern: {path_or_pattern}")
        return shards

def decode_png_rgb(png_bytes) -> np.ndarray:
    """bytes -> uint8 RGB [H,W,3]"""
    arr = np.frombuffer(png_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode PNG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def decode_npy(npy_bytes) -> np.ndarray:
    """bytes -> np.ndarray via np.load on BytesIO"""
    with io.BytesIO(npy_bytes) as f:
        return np.load(f, allow_pickle=False)

def parse_key(key: str):
    """Split __key__ like 'run1_000000123' -> ('run1', 123)"""
    if "_" not in key:
        return key, -1
    sess, idx = key.split("_", 1)
    try:
        idxi = int(idx)
    except Exception:
        idxi = -1
    return sess, idxi


# --------------------------------
# WebDataset → sequence dataset
# --------------------------------

class SeqFromWDS(IterableDataset):
    """
    Stream WebDataset samples, build sliding sequences per session.

    Yields:
      X: [T, H, W, 5] float32  (train RGB [0..1], elev (meters), mask M∈{0,1})
      Y: [H, W, 4]   float32   (GT RGB [0..1], GT elev (meters))
    """
    def __init__(self, shards_pattern: str, seq_len: int = 4, shuffle_shards: bool = True,
                 shuffle_samples: bool = False, buffer_size: int = 0):
        super().__init__()
        self.pattern = shards_pattern
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.buffer_size = buffer_size  # optional per-session buffer limit

    def __iter__(self):
        # Build the base webdataset pipeline (accept directory or pattern)
        shard_paths = resolve_shards(self.pattern)
        dataset = wds.WebDataset(
            shard_paths,
            shardshuffle=self.shuffle_shards,
            empty_check=False
        )
        # Compatibility: older WebDataset versions don't have .split_by_node/worker methods
        if hasattr(wds, 'split_by_node'):
            dataset = wds.split_by_node(dataset)
        if hasattr(wds, 'split_by_worker'):
            dataset = wds.split_by_worker(dataset)
        # Some WebDataset versions return a generator after split_*; guard .with_length
        if hasattr(dataset, 'with_length'):
            dataset = dataset.with_length(None)
        for sample in dataset:
            # Expect keys to exist
            key = sample.get("__key__")
            if key is None:
                continue
            sess, idx = parse_key(key)

            # Decode pieces
            tr_rgb = decode_png_rgb(sample["tr_rgb.png"]).astype(np.float32) / 255.0
            gt_rgb = decode_png_rgb(sample["gt_rgb.png"]).astype(np.float32) / 255.0

            tr_elev = decode_npy(sample["tr_elev.npy"]).astype(np.float32)
            gt_elev = decode_npy(sample["gt_elev.npy"]).astype(np.float32)

            H, W = tr_elev.shape
            # Training valid mask (from training elevation finite)
            # Maintain per-session history buffer
            tr_mask = np.isfinite(tr_elev).astype(np.float32)

            if "valid.png" in sample:
                valid_img = cv2.imdecode(np.frombuffer(sample["valid.png"], np.uint8), cv2.IMREAD_GRAYSCALE)
                gt_mask = (valid_img > 0).astype(np.float32)
            else:
                gt_mask = np.isfinite(gt_elev).astype(np.float32)

            # >>> sanitize elevations <<<
            tr_elev_clean = np.nan_to_num(tr_elev, nan=0.0, posinf=0.0, neginf=0.0)

            # keep gt_mask as the source of truth; fill only where invalid
            gt_elev_clean = np.where(gt_mask > 0.5, gt_elev, 0.0).astype(np.float32)
            
            if not hasattr(self, "_hist"):
                self._hist = defaultdict(lambda: deque(maxlen=max(self.seq_len, self.buffer_size or self.seq_len)))
            # use the cleaned arrays below
            self._hist[sess].append({
                "tr_rgb": tr_rgb,
                "tr_elev": tr_elev_clean,   # <—
                "tr_mask": tr_mask,
                "gt_rgb": gt_rgb,
                "gt_elev": gt_elev_clean,   # <—
                "gt_mask": gt_mask
            })

            if len(self._hist[sess]) >= self.seq_len:
                # Build sequence X from the last T training frames
                frames = list(self._hist[sess])[-self.seq_len:]
                X = []
                for fr in frames:
                    # stack [R,G,B,elev,M]
                    x = np.concatenate(
                        [fr["tr_rgb"], fr["tr_elev"][..., None], fr["tr_mask"][..., None]], axis=-1
                    )
                    X.append(x)
                X = np.stack(X, axis=0)  # [T,H,W,5]

                # Target = GT of the last frame
                Y = np.concatenate(
                    [frames[-1]["gt_rgb"], frames[-1]["gt_elev"][..., None]], axis=-1
                )  # [H,W,4]

                # Also hand back GT-valid mask for loss
                Mgt = frames[-1]["gt_mask"].astype(np.float32)  # [H,W]

                yield X, Y, Mgt


# ----------------------------
# Model: U-Net + ConvLSTM
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch, out_ch)  # concat skip => in_ch = out_ch(up) + skip
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        if x.shape[-2:] != skip.shape[-2:]:
            diffY = skip.shape[-2] - x.shape[-2]
            diffX = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, k=3):
        super().__init__()
        p = k // 2
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, k, padding=p)

    def forward(self, x, h, c):
        # x: [B,in_ch,H,W]; h,c: [B,hidden_ch,H,W]
        if h is None:
            B, _, H, W = x.shape
            h = torch.zeros(B, self.hidden_ch, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_ch, H, W, device=x.device)
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(z, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden_ch)

    def forward(self, x_seq):
        """
        x_seq: [B,T,C,H,W]  -> returns last hidden [B,hidden_ch,H,W]
        """
        h = c = None
        for t in range(x_seq.shape[1]):
            h, c = self.cell(x_seq[:, t], h, c)
        return h  # last hidden

class UNetConvLSTM(nn.Module):
    def __init__(self, in_ch=5, base=32, out_ch=4):
        super().__init__()
        # Encoder (shared for each time step)
        self.enc1 = ConvBlock(in_ch, base)        # -> base
        self.enc2 = Down(base, base * 2)          # -> 2b
        self.enc3 = Down(base * 2, base * 4)      # -> 4b

        # ConvLSTM on bottleneck features
        self.lstm = ConvLSTM(in_ch=base * 4, hidden_ch=base * 4)

        # Decoder (use last frame skips)
        self.up2 = Up(base * 4, base * 2)
        self.up1 = Up(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        """
        x: [B,T,C,H,W]
        returns: [B,4,H,W] with channels [R,G,B,elev]; RGB should be in [0,1].
        """
        B, T, C, H, W = x.shape

        # Encode each frame; stash last frame's skips
        feats3_seq = []
        for t in range(T):
            x_t = x[:, t]                      # [B,C,H,W]
            f1 = self.enc1(x_t)                # [B,b,H,W]
            f2 = self.enc2(f1)                 # [B,2b,H/2,W/2]
            f3 = self.enc3(f2)                 # [B,4b,H/4,W/4]
            feats3_seq.append(f3)
            if t == T - 1:
                skip2, skip1 = f2, f1

        feats3 = torch.stack(feats3_seq, dim=1)     # [B,T,4b,H/4,W/4]
        bottleneck = self.lstm(feats3)              # [B,4b,H/4,W/4]

        d2 = self.up2(bottleneck, skip2)            # [B,2b,H/2,W/2]
        d1 = self.up1(d2, skip1)                    # [B,b,H,W]
        out = self.out(d1)                          # [B,4,H,W]

        # Bound RGB to [0,1] via sigmoid; keep elevation linear
        rgb = torch.sigmoid(out[:, :3])
        elev = out[:, 3:4]
        return torch.cat([rgb, elev], dim=1)


# ----------------------------
# Training loop
# ----------------------------

def masked_l1(pred, target, mask, eps=1e-6):
    """
    pred,target: [B,C,H,W], mask: [B,1,H,W] with 1=valid
    """
    diff = torch.abs(pred - target) * mask
    denom = mask.sum() + eps
    return diff.sum() / denom

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    ds = SeqFromWDS(args.shards, seq_len=args.seq_len, shuffle_shards=True)
    # Collate: stack tensors
    def _collate(batch):
        Xs, Ys, Ms = [], [], []
        for X, Y, M in batch:
            # X: [T,H,W,5] -> [T,5,H,W]
            Xs.append(torch.from_numpy(np.transpose(X, (0,3,1,2))).float())
            Ys.append(torch.from_numpy(np.transpose(Y, (2,0,1))).float())  # [4,H,W]
            Ms.append(torch.from_numpy(M[None, ...]).float())               # [1,H,W]
        return torch.stack(Xs), torch.stack(Ys), torch.stack(Ms)

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, collate_fn=_collate)

    model = UNetConvLSTM(in_ch=5, base=args.base, out_ch=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Simple cosine decay (optional)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    os.makedirs(args.out, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"rgb": 0.0, "elev": 0.0, "total": 0.0}
        for step, (X, Y, M) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}", ncols=100), start=1):
            # Shapes
            # X: [B,T,5,H,W], Y: [B,4,H,W], M: [B,1,H,W]
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)

            # Split targets
            Y_rgb = Y[:, :3]        # [0..1]
            Y_elev = Y[:, 3:4]

            # Forward
            preds = model(X)        # [B,4,H,W]
            P_rgb = preds[:, :3]
            P_elev = preds[:, 3:4]

            # Losses (masked by GT-valid)
            w_mask = M              # same mask for RGB and elev here
            loss_rgb = masked_l1(P_rgb, Y_rgb, w_mask)
            loss_elev = masked_l1(P_elev, Y_elev, w_mask)

            loss = args.w_rgb * loss_rgb + args.w_elev * loss_elev

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["rgb"] += loss_rgb.item()
            running["elev"] += loss_elev.item()
            running["total"] += loss.item()
            global_step += 1

            if step % args.log_every == 0:
                n = args.log_every
                print(f"[epoch {epoch:03d} step {step:05d}] "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"rgb={running['rgb']/n:.4f} elev={running['elev']/n:.4f} "
                      f"loss={running['total']/n:.4f}")
                running = {"rgb": 0.0, "elev": 0.0, "total": 0.0}

        sched.step()

        # Save checkpoint each epoch
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out, f"checkpoint_epoch_{epoch:03d}.pt"))

    print("Training complete.")


# ----------------------------
# CLI
# ----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Train ConvLSTM U-Net on GridMap WebDataset.")
    p.add_argument("--shards", required=True,
                   help="Path to directory containing .tar shards OR a pattern (e.g., '/data/shard-*.tar')")
    p.add_argument("--out", default="./runs/gridmap_convLSTM", help="Output directory for checkpoints")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4, help="Temporal context length T")
    p.add_argument("--base", type=int, default=32, help="UNet base channels")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-rgb", type=float, default=1.0, help="Weight for RGB L1")
    p.add_argument("--w-elev", type=float, default=1.0, help="Weight for elevation L1")
    p.add_argument("--log-every", type=int, default=50)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
# python3 train.py --shards /media/slsecret/T7/carla3/data/town1/gridmap_wds --workers 4
