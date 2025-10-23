#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import io
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import webdataset as wds
from tqdm import tqdm

# --- Crucially, import the LUT and original class names ---
from color_mapping import ORIG, RGB_TO_ORIG_ID_LUT

def decode_png_rgb(png_bytes: bytes) -> np.ndarray:
    """
    Robustly decodes PNG bytes to a 3-channel (H, W, 3) RGB numpy array.
    """
    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode PNG image")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported image format with shape: {img.shape}")

def calculate_original_class_weights(shard_paths, num_workers=4):
    """
    Reads original shards directly and correctly calculates class weights.
    """
    print(f"Processing {len(shard_paths)} original shards to calculate class weights...")

    dataset = wds.WebDataset(shard_paths, handler=wds.warn_and_continue)
    # Using batch_size=None is correct for sample-wise iteration
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)

    class_counts = torch.zeros(len(ORIG), dtype=torch.int64)
    lut_tensor = torch.from_numpy(RGB_TO_ORIG_ID_LUT).long()

    pbar = tqdm(loader, desc="Counting original class pixels", ncols=100)
    for sample in pbar:
        if "gt_rgb.png" not in sample:
            continue

        rgb_tensor = None # Define for the except block
        try:
            rgb_img = decode_png_rgb(sample["gt_rgb.png"])
            rgb_tensor = torch.from_numpy(rgb_img)

            # --- THE FIX IS HERE ---
            # Convert r, g, b to .long() for indexing
            r = rgb_tensor[..., 0].long()
            g = rgb_tensor[..., 1].long()
            b = rgb_tensor[..., 2].long()
            
            # Now indexing will work correctly
            original_ids = lut_tensor[r, g, b]

            valid_labels = original_ids[original_ids > 0]
            if valid_labels.numel() > 0:
                counts = torch.bincount(valid_labels, minlength=len(ORIG))
                class_counts += counts

        except Exception as e:
            key = sample.get('__key__', 'unknown')
            shape_info = f"with shape {rgb_tensor.shape}" if rgb_tensor is not None else ""
            print(f"\n[Warning] Skipping sample {key} {shape_info} due to error: {e}")

    print("\n--- Raw Pixel Counts (Original 29-Class Mapping) ---")
    for i, count in enumerate(class_counts):
        print(f"Class {i:02d} ({ORIG[i][1]}): {count.item():,}")

    weights = torch.zeros(len(ORIG), dtype=torch.float32)
    valid_counts = class_counts[1:]
    present_classes_mask = valid_counts > 0
    if present_classes_mask.any():
        log_weights = 1.0 / torch.log(1.02 + valid_counts[present_classes_mask])
        log_weights = (log_weights / log_weights.mean())
        weights[1:][present_classes_mask] = log_weights

    for i in range(1, len(ORIG)):
        if class_counts[i] == 0:
            print(f"-> WARNING: Class {i} ({ORIG[i][1]}) has 0 pixels. Its weight is set to 0.")

    print("\n--- Calculated Class Weights (Log-Normalized) ---")
    for i, w in enumerate(weights):
        print(f"Class {i:02d} ({ORIG[i][1]}): {w.item():.4f}")

    return weights.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate class weights from original dataset shards.")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers (default: 4).")
    args = parser.parse_args()

    train_dir = os.path.join("/media/slsecret/T7/carla3/data/town7/", "gridmap_wds")
    
    from shard_utils import resolve_shards
    train_shards = resolve_shards(train_dir, recursive=False, only_remapped=False)
    
    class_weights = calculate_original_class_weights(train_shards, args.workers)
    
    print("\nFinal Class Weights List:")
    print(class_weights)