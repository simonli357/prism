#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates the quality of *input* data (tr_elev, tr_inpaint, tr_onehot14)
against the *ground truth* data (gt_elev, gt_onehot14) from the 
test set shards.

Generates CSV files (per-frame and overall summary) and boxplots 
for the computed metrics.
"""

import os
import glob
import io
import argparse
import numpy as np
import webdataset as wds
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import metrics from the standalone script
try:
    from metrics import (compute_elevation_metrics, 
                         compute_miou, 
                         compute_bf_score)
except ImportError:
    print("Error: Could not import from metrics.py.")
    print("Please ensure metrics.py is in the same directory or in your PYTHONPATH.")
    exit(1)

# --- Utility functions (from shard_utils.py) ---

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

def decode_npy(npy_bytes) -> np.ndarray:
    """Decodes .npy file bytes into a numpy array."""
    with io.BytesIO(npy_bytes) as f:
        return np.load(f, allow_pickle=False)

# --- Main Evaluation Logic ---

def main(args):
    """
    Main function to process shards and compute metrics.
    """
    print(f"Starting evaluation of test set inputs from: {args.shard_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Find and load shards
    try:
        shard_paths = resolve_shards(args.shard_dir)
    except FileNotFoundError as e:
        print(e)
        return
        
    dataset = wds.WebDataset(shard_paths)
    
    results_elev = []
    results_sem = []
    
    # 2. Process each sample
    print("Processing samples...")
    for sample in tqdm(dataset):
        key = sample.get("__key__")
        if key is None:
            continue
            
        try:
            # Load elevation data
            gt_elev = decode_npy(sample["gt_elev.npy"])
            tr_elev = decode_npy(sample["tr_elev.npy"])
            tr_inpaint = decode_npy(sample["tr_inpaint.npy"])
            
            # Load semantic data
            gt_sem_onehot = decode_npy(sample["gt_onehot14.npy"])
            tr_sem_onehot = decode_npy(sample["tr_onehot14.npy"])

            # --- Process Elevation ---
            
            # Use finiteness of GT elevation as the primary valid mask
            gt_valid_mask = np.isfinite(gt_elev)
            
            # Process GT elev: set invalid pixels to 0
            gt_elev_proc = np.where(gt_valid_mask, gt_elev, 0.0)
            
            # Process Training elev: set nan/inf to 0
            tr_elev_proc = np.nan_to_num(tr_elev, nan=0.0, posinf=0.0, neginf=0.0)
            tr_inpaint_proc = np.nan_to_num(tr_inpaint, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Metric 1: Compare raw sparse input (tr_elev) to GT
            # Mask: only where raw input *and* GT are valid
            sparse_mask = np.isfinite(tr_elev) & gt_valid_mask
            elev_metrics_raw = compute_elevation_metrics(
                tr_elev_proc, gt_elev_proc, sparse_mask
            )
            
            # Metric 2: Compare inpainted input (tr_inpaint) to GT
            # Mask: only where GT is valid
            elev_metrics_inpaint = compute_elevation_metrics(
                tr_inpaint_proc, gt_elev_proc, gt_valid_mask
            )
            
            results_elev.append({
                'key': key,
                'mae_raw_vs_gt': elev_metrics_raw['mae'],
                'rmse_raw_vs_gt': elev_metrics_raw['rmse'],
                'mae_inpaint_vs_gt': elev_metrics_inpaint['mae'],
                'rmse_inpaint_vs_gt': elev_metrics_inpaint['rmse'],
            })

            # --- Process Semantics ---
            
            # Convert one-hot to integer labels
            gt_sem_labels = np.argmax(gt_sem_onehot, axis=-1)
            tr_sem_labels = np.argmax(tr_sem_onehot, axis=-1)
            
            # Use the GT elevation mask to find invalid pixels
            # Set invalid pixels to -1 (ignore_label)
            ignore_label = -1
            gt_sem_labels_masked = np.where(gt_valid_mask, gt_sem_labels, ignore_label)
            tr_sem_labels_masked = np.where(gt_valid_mask, tr_sem_labels, ignore_label)

            # Metric 3: mIoU for input vs. GT
            sem_miou = compute_miou(
                tr_sem_labels_masked, 
                gt_sem_labels_masked, 
                args.num_classes, 
                ignore_label=ignore_label
            )
            
            # Metric 4: BF-Score for input vs. GT
            sem_bf = compute_bf_score(
                tr_sem_labels_masked, 
                gt_sem_labels_masked,
                ignore_label=ignore_label
            )
            
            results_sem.append({
                'key': key,
                'miou': sem_miou['miou'],
                'bf_score': sem_bf['bf_score'],
            })

        except Exception as e:
            print(f"Warning: Failed to process sample {key}. Error: {e}")
            continue
            
    # 3. Aggregate results and save to CSV
    if not results_elev or not results_sem:
        print("No results were processed. Exiting.")
        return

    print("\n--- Aggregating and Saving Results ---")
    
    # --- Per-frame CSVs ---
    # Elevation
    df_elev = pd.DataFrame(results_elev)
    csv_elev_path = os.path.join(args.output_dir, 'elevation_input_metrics_per_frame.csv')
    df_elev.to_csv(csv_elev_path, index=False)
    print(f"Saved per-frame elevation metrics to: {csv_elev_path}")
    print("Mean Elevation Input Metrics:")
    mean_elev_metrics = df_elev.mean(numeric_only=True)
    print(mean_elev_metrics)
    
    # Semantics
    df_sem = pd.DataFrame(results_sem)
    csv_sem_path = os.path.join(args.output_dir, 'semantic_input_metrics_per_frame.csv')
    df_sem.to_csv(csv_sem_path, index=False)
    print(f"Saved per-frame semantic metrics to: {csv_sem_path}")
    print("\nMean Semantic Input Metrics:")
    mean_sem_metrics = df_sem.mean(numeric_only=True)
    print(mean_sem_metrics)
    
    # --- NEW: Create and Save Overall Summary CSV ---
    print("\nSaving overall summary...")
    
    # Combine the mean metrics into one series/dataframe
    summary_metrics = pd.concat([mean_elev_metrics, mean_sem_metrics])
    summary_df = summary_metrics.to_frame().T
    summary_df.index = ['mean_score']
    
    csv_summary_path = os.path.join(args.output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(csv_summary_path)
    
    print(f"Saved overall summary metrics to: {csv_summary_path}")
    print(summary_df)

    
    # 4. Generate Plots
    print("\nGenerating plots...")
    
    # Elevation Plot
    try:
        df_elev_melted = df_elev.melt(id_vars=['key'], var_name='Metric', value_name='Error')
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df_elev_melted, x='Metric', y='Error')
        plt.title('Elevation Input Error (vs. Ground Truth)')
        plt.ylabel('Error (meters)')
        plt.xlabel('Metric (Input Type vs. Ground Truth)')
        plt.xticks(rotation=15)
        plot_elev_path = os.path.join(args.output_dir, 'elevation_metrics_boxplot.png')
        plt.savefig(plot_elev_path, bbox_inches='tight')
        plt.close()
        print(f"Saved elevation plot to: {plot_elev_path}")

        # Semantic Plot
        df_sem_melted = df_sem.melt(id_vars=['key'], var_name='Metric', value_name='Score')
        plt.figure(figsize=(10, 7))
        sns.boxplot(data=df_sem_melted, x='Metric', y='Score')
        plt.title('Semantic Input Score (vs. Ground Truth)')
        plt.ylabel('Score')
        plt.xlabel('Metric')
        plt.ylim(0, 1)
        plot_sem_path = os.path.join(args.output_dir, 'semantic_metrics_boxplot.png')
        plt.savefig(plot_sem_path, bbox_inches='tight')
        plt.close()
        print(f"Saved semantic plot to: {plot_sem_path}")
        
    except Exception as e:
        print(f"Warning: Could not generate plots. Error: {e}")
        print("This may be due to missing libraries (seaborn, matplotlib) or an issue with the data.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate test set input data against ground truth."
    )
    parser.add_argument(
        "--shard_dir", 
        type=str, 
        default="/media/slsecret/T7/carla3/data_split357/test",
        help="Directory or glob pattern for test set .tar shards."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./test_set_input_evaluation",
        help="Directory to save CSVs and plots."
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=14,
        help="Number of semantic classes."
    )
    
    args = parser.parse_args()
    main(args)