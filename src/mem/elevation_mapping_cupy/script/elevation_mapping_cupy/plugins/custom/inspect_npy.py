#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspects and compares the statistics of gt_elev, tr_elev, and a model's
pred_elev for a single frame to understand their properties.
"""

import os
import glob
import io
import argparse
import numpy as np
import webdataset as wds
from tqdm import tqdm

# --- WDS Utilities ---

def decode_npy(npy_bytes) -> np.ndarray:
    """Decodes .npy file bytes into a numpy array."""
    with io.BytesIO(npy_bytes) as f:
        return np.load(f, allow_pickle=False)

def resolve_shards(path_or_pattern: str):
    """Finds all .tar shards given a directory or glob pattern."""
    paths = []
    if os.path.isdir(path_or_pattern):
        paths = sorted(glob.glob(os.path.join(path_or_pattern, "*.tar")))
    else:
        paths = sorted(glob.glob(path_or_pattern))
    
    if not paths:
        raise FileNotFoundError(f"No .tar shards found using: {path_or_pattern}")
    return paths

# --- Analysis Function ---

def analyze_array(name: str, arr: np.ndarray):
    """Prints detailed statistics about a numpy array."""
    
    print("-" * 30)
    print(f"Analysis for: {name}")
    print(f"Shape: {arr.shape}")
    
    total_pixels = arr.size
    if total_pixels == 0:
        print("Array is empty.")
        return

    finite_mask = np.isfinite(arr)
    finite_count = np.sum(finite_mask)
    nan_count = total_pixels - finite_count
    
    print(f"Total Pixels: {total_pixels}")
    print(f"Finite Pixels: {finite_count} ({finite_count / total_pixels * 100:.2f}%)")
    print(f"NaN/Inf Pixels: {nan_count} ({nan_count / total_pixels * 100:.2f}%)")
    
    if finite_count > 0:
        valid_data = arr[finite_mask]
        zero_count = np.sum(valid_data == 0.0)
        
        print("\nStatistics (for FINITE pixels):")
        print(f"  Mean:   {np.mean(valid_data):.4f}")
        print(f"  Median: {np.median(valid_data):.4f}")
        print(f"  Std Dev: {np.std(valid_data):.4f}")
        print(f"  Min:    {np.min(valid_data):.4f}")
        print(f"  Max:    {np.max(valid_data):.4f}")
        print(f"  Zeroes: {zero_count} ({zero_count / finite_count * 100:.2f}% of finite)")
    else:
        print("\nArray has no finite data.")
    print("-" * 30)

# --- Main Logic ---

def main(args):
    
    gt_arr, tr_arr, key = None, None, None
    
    # 1. Find and load the ground truth sample
    print(f"Searching for sample '{args.key}' in {args.gt_shard_dir}...")
    try:
        shard_paths = resolve_shards(args.gt_shard_dir)
    except FileNotFoundError as e:
        print(e)
        return
        
    dataset = wds.WebDataset(shard_paths)
    
    for sample in tqdm(dataset, desc="Scanning GT shards"):
        if sample.get("__key__") == args.key:
            try:
                gt_arr = decode_npy(sample["gt_elev.npy"])
                tr_arr = decode_npy(sample["tr_elev.npy"])
                key = sample.get("__key__")
                break
            except Exception as e:
                print(f"Error decoding sample {args.key}: {e}")
                return
    
    if key is None:
        print(f"Error: Could not find sample with key '{args.key}' in {args.gt_shard_dir}")
        print("Please check the key name (e.g., 'run1_000000300')")
        return

    print(f"Found and loaded sample {key}.")

    # 2. Find and load the prediction
    pred_arr = None
    if args.model_name:
        pred_pattern = os.path.join(
            args.model_runs_dir,
            args.model_name,
            "inference_on_test_set",
            "**",
            f"{key}.pred_elev.npy"
        )
        
        print(f"Searching for prediction file: {pred_pattern}")
        pred_files = glob.glob(pred_pattern, recursive=True)
        
        if not pred_files:
            print(f"Error: Could not find prediction for {key} for model {args.model_name}")
        else:
            try:
                pred_arr = np.load(pred_files[0])
                print(f"Loaded prediction from: {pred_files[0]}")
            except Exception as e:
                print(f"Error loading prediction file {pred_files[0]}: {e}")

    # 3. Analyze and Print
    analyze_array(f"Ground Truth ({key}.gt_elev.npy)", gt_arr)
    analyze_array(f"Training Input ({key}.tr_elev.npy)", tr_arr)
    if pred_arr is not None:
        analyze_array(f"Prediction ({args.model_name})", pred_arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect and compare statistics of elevation .npy files."
    )
    parser.add_argument(
        "--gt_shard_dir", 
        type=str, 
        default="/media/slsecret/T7/carla3/data_split357/test",
        help="Directory or glob pattern for ground truth .tar shards."
    )
    parser.add_argument(
        "--model_runs_dir", 
        type=str, 
        default="/media/slsecret/T7/carla3/runs",
        help="Base directory containing all model run folders."
    )
    parser.add_argument(
        "--key", 
        type=str, 
        default="run1_000000300",
        help="The specific sample key to inspect (e.g., 'run1_000000300')."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="all357_unet",
        help="Model to inspect (e.g., 'all357_unet' or 'all357_cnn'). Set to 'none' to skip."
    )
    
    args = parser.parse_args()
    
    if args.model_name.lower() == 'none':
        args.model_name = None
        
    main(args)