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

try:
    from .color_mapping import color28_to_onehot14, color28_to_color14, onehot14_to_color, SEM_CHANNELS
except ImportError:
    from color_mapping import color28_to_onehot14, color28_to_color14, onehot14_to_color, SEM_CHANNELS


def resolve_shards(path_or_pattern: str, recursive: bool = False, only_remapped: bool = False):
    """
    Return a sorted list of .tar shard files.
    - If `path_or_pattern` is a directory: walk it (recursively if requested).
    - If it's a glob pattern: expand it (non-recursive glob).
    - If `only_remapped` is True: keep shards whose *directory path* contains "remapped".
    """
    paths = []
    if os.path.isdir(path_or_pattern):
        if recursive:
            for dirpath, _, filenames in os.walk(path_or_pattern):
                if only_remapped and ("remapped" not in dirpath):
                    continue
                for fn in filenames:
                    if fn.endswith(".tar"):
                        paths.append(os.path.join(dirpath, fn))
        else:
            candidates = glob.glob(os.path.join(path_or_pattern, "*.tar"))
            if only_remapped:
                candidates = [p for p in candidates if "remapped" in os.path.dirname(p)]
            paths = sorted(candidates)
    else:
        paths = sorted(glob.glob(path_or_pattern))
        if only_remapped:
            paths = [p for p in paths if "remapped" in os.path.dirname(p)]

    if not paths:
        raise FileNotFoundError(f"No .tar shards found using: {path_or_pattern} "
                                f"(recursive={recursive}, only_remapped={only_remapped})")
    print(f"[INFO] Found {len(paths)} shard(s)")
    return paths

def decode_png_rgb(png_bytes) -> np.ndarray:
    """bytes -> uint8 RGB [H,W,3]"""
    arr = np.frombuffer(png_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Failed to decode PNG")
    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    elif bgr.shape[2] == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    return rgb

def decode_npy(npy_bytes) -> np.ndarray:
    with io.BytesIO(npy_bytes) as f:
        return np.load(f, allow_pickle=False)

def parse_key(key: str):
    if "_" not in key: return key, -1
    sess, idx = key.split("_", 1)
    try: idxi = int(idx)
    except Exception: idxi = -1
    return sess, idxi

class SeqFromWDS(IterableDataset):
    def __init__(self, shard_paths: list, seq_len: int = 4,
                 shuffle_shards: bool = True,
                 include_mask: bool = True):
        super().__init__()
        self.shard_paths = shard_paths
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.include_mask = include_mask

    def __iter__(self):
        dataset = wds.WebDataset(self.shard_paths, shardshuffle=self.shuffle_shards, empty_check=False)
        if hasattr(wds, 'split_by_node'):   dataset = wds.split_by_node(dataset)
        if hasattr(wds, 'split_by_worker'): dataset = wds.split_by_worker(dataset)
        if hasattr(dataset, 'with_length'): dataset = dataset.with_length(None)

        hist = defaultdict(lambda: deque(maxlen=self.seq_len))

        for sample in dataset:
            key = sample.get("__key__")
            if key is None:
                continue
            sess, idx = parse_key(key)

            if "tr_onehot14.npy" in sample:
                tr_sem = decode_npy(sample["tr_onehot14.npy"]).astype(np.float32)
            else:
                tr_rgb28 = decode_png_rgb(sample["tr_rgb.png"])
                tr_sem = color28_to_onehot14(tr_rgb28, dtype=np.float32)
            if tr_sem.ndim != 3 or tr_sem.shape[2] != SEM_CHANNELS:
                raise ValueError(f"tr_sem shape {tr_sem.shape} != (H,W,14)")

            if "gt_onehot14.npy" in sample:
                gt_sem = decode_npy(sample["gt_onehot14.npy"]).astype(np.float32)
            else:
                gt_rgb28 = decode_png_rgb(sample["gt_rgb.png"])
                gt_sem = color28_to_onehot14(gt_rgb28, dtype=np.float32)
            if gt_sem.ndim != 3 or gt_sem.shape[2] != SEM_CHANNELS:
                raise ValueError(f"gt_sem shape {gt_sem.shape} != (H,W,14)")

            tr_elev = decode_npy(sample["tr_elev.npy"]).astype(np.float32)
            gt_elev = decode_npy(sample["gt_elev.npy"]).astype(np.float32)

            tr_mask = np.isfinite(tr_elev).astype(np.float32)
            if "valid.png" in sample:
                valid_img = cv2.imdecode(np.frombuffer(sample["valid.png"], np.uint8), cv2.IMREAD_GRAYSCALE)
                gt_mask = (valid_img > 0).astype(np.float32)
            else:
                gt_mask = np.isfinite(gt_elev).astype(np.float32)

            tr_elev = np.nan_to_num(tr_elev, nan=0.0, posinf=0.0, neginf=0.0)
            gt_elev = np.where(gt_mask > 0.5, gt_elev, 0.0).astype(np.float32)

            hist[sess].append({
                "tr_sem": tr_sem,
                "tr_elev": tr_elev,
                "tr_mask": tr_mask,
                "gt_sem": gt_sem,
                "gt_elev": gt_elev,
                "gt_mask": gt_mask,
                "key": key,
            })

            if len(hist[sess]) < self.seq_len:
                continue

            frames = list(hist[sess])[-self.seq_len:]

            X = []
            for fr in frames:
                parts = [fr["tr_sem"], fr["tr_elev"][..., None]]
                if self.include_mask:
                    parts.append(fr["tr_mask"][..., None])
                X.append(np.concatenate(parts, axis=-1))
            X = np.stack(X, axis=0)  # [T,H,W,C_in]

            last = frames[-1]
            Mgt = last["gt_mask"].astype(np.float32)
            
            # Y = np.concatenate([last["gt_sem"], last["gt_elev"][..., None]], axis=-1)  # [H,W,15]
            # yield X, Y, Mgt
            
            gt_sem_indices = np.argmax(last["gt_sem"], axis=-1).astype(np.int64) # [H,W]
            gt_elev_target = last["gt_elev"][..., None] # [H,W,1]
            Y_sem = gt_sem_indices
            Y_elev = gt_elev_target
            yield X, (Y_sem, Y_elev), Mgt