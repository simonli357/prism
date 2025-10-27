#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import io
import argparse
import numpy as np
import cv2
import webdataset as wds
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

try:
    from metrics import (compute_elevation_metrics, 
                         compute_miou, 
                         compute_bf_score)
except ImportError:
    print("Error: Could not import from metrics.py.")
    print("Please ensure metrics.py is in the same directory or in your PYTHONPATH.")
    exit(1)

SEM_CHANNELS = 12

NEW_PALETTE = np.array([
    (0,   0,   0),     # 0 Unlabeled
    (128, 64,  128),   # 1 Roads
    (244, 35,  232),   # 2 SideWalks
    (70,  70,  70),    # 3 Structure
    (180, 165, 180),   # 4 Barrier
    (220, 220, 0),     # 5 PoleSign
    (107, 142, 35),    # 6 Vegetation
    (152, 251, 152),   # 7 Terrain
    (220, 20,  60),    # 8 Person
    (119, 11,  32),    # 9 TwoWheeler
    (0,   0,   142),   # 10 Vehicle
    (45,  60,  150),   # 11 Water
], dtype=np.uint8)

def _build_reverse_color_lut():
    lut = np.zeros((256, 256, 256), dtype=np.int32)
    for i, (r, g, b) in enumerate(NEW_PALETTE):
        lut[r, g, b] = i
    return lut

_NEW_RGB_TO_ID_LUT = _build_reverse_color_lut()

def color12_to_labels(rgb_img):
    if rgb_img.dtype != np.uint8:
        rgb_img = rgb_img.astype(np.uint8)
    return _NEW_RGB_TO_ID_LUT[rgb_img[..., 0], rgb_img[..., 1], rgb_img[..., 2]]

def decode_npy(npy_bytes) -> np.ndarray:
    with io.BytesIO(npy_bytes) as f:
        return np.load(f, allow_pickle=False)

def decode_png_rgb(png_bytes) -> np.ndarray:
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

def resolve_shards(path_or_pattern: str):
    paths = []
    if os.path.isdir(path_or_pattern):
        paths = sorted(glob.glob(os.path.join(path_or_pattern, "*.tar")))
    else:
        paths = sorted(glob.glob(path_or_pattern))
    
    if not paths:
        raise FileNotFoundError(f"No .tar shards found using: {path_or_pattern}")
    print(f"[INFO] Found {len(paths)} ground truth shard(s)")
    return paths

def build_prediction_map(model_runs_dir: str, model_names: list) -> dict:
    print(f"[INFO] Building prediction map from: {model_runs_dir}")
    pred_map = defaultdict(dict)
    
    for model in model_names:
        inference_dir = os.path.join(model_runs_dir, model, "inference_on_test_set")
        if not os.path.isdir(inference_dir):
            print(f"Warning: Inference dir not found for model '{model}'. Skipping.")
            continue

        elev_files = glob.glob(os.path.join(inference_dir, "**", "*.pred_elev.npy"), recursive=True)
        sem_files = glob.glob(os.path.join(inference_dir, "**", "*.pred_color14.png"), recursive=True)
        
        for f in elev_files:
            key = os.path.basename(f).split('.')[0]
            if key not in pred_map[model]:
                pred_map[model][key] = {}
            pred_map[model][key]['elev_path'] = f
            
        for f in sem_files:
            key = os.path.basename(f).split('.')[0]
            if key not in pred_map[model]:
                pred_map[model][key] = {}
            pred_map[model][key]['sem_path'] = f
        
        print(f"  > Found {len(pred_map[model])} prediction samples for model '{model}'")
        
    return pred_map

def build_result_row(key: str, model_name: str, elev_metrics: dict, 
                     sem_miou: dict, sem_bf: dict, num_classes: int,
                     gt_stats: dict, pred_stats: dict) -> dict:
    row = {
        'key': key,
        'model_name': model_name,
        'mae': elev_metrics.get('mae', np.nan),
        'rmse': elev_metrics.get('rmse', np.nan),
        'miou': sem_miou.get('miou', np.nan),
        'bf_score': sem_bf.get('bf_score', np.nan),
        
        'gt_elev_mean': gt_stats.get('mean', np.nan),
        'gt_elev_median': gt_stats.get('median', np.nan),
        'gt_elev_std': gt_stats.get('std', np.nan),
        
        'pred_elev_mean': pred_stats.get('mean', np.nan),
        'pred_elev_median': pred_stats.get('median', np.nan),
        'pred_elev_std': pred_stats.get('std', np.nan),
    }
    
    iou_per_class = sem_miou.get('iou_per_class', np.full(num_classes, np.nan))
    for i in range(num_classes):
        row[f'iou_class_{i}'] = iou_per_class[i]
        
    return row

def get_elevation_stats(elevation_map: np.ndarray, mask: np.ndarray) -> dict:
    """Calculates stats for an elevation map within a given mask."""
    if mask.sum() == 0:
        return {'mean': np.nan, 'median': np.nan, 'std': np.nan}
        
    valid_elev = elevation_map[mask > 0.5]
    
    return {
        'mean': np.mean(valid_elev),
        'median': np.median(valid_elev),
        'std': np.std(valid_elev)
    }

def main(args):
    num_classes = SEM_CHANNELS
    print(f"[INFO] Using {num_classes} semantic classes.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_names = [d for d in os.listdir(args.model_runs_dir) 
                   if os.path.isdir(os.path.join(args.model_runs_dir, d))]
    pred_map = build_prediction_map(args.model_runs_dir, model_names)

    try:
        shard_paths = resolve_shards(args.gt_shard_dir)
    except FileNotFoundError as e:
        print(e)
        return
        
    dataset = wds.WebDataset(shard_paths)
    all_results = []
    
    sem_ignore_label = 0 
    invalid_elev_val = -100.0  # <<<--- YOUR CHANGE IS HERE
    
    print(f"[INFO] Using semantic ignore_label={sem_ignore_label}")
    print(f"[INFO] Using invalid elevation value={invalid_elev_val}")
    print(f"[INFO] Using BF-Score tolerance={args.bf_tol}px")
    print(f"[INFO] Using Elevation MAE outlier threshold={args.elevation_mae_threshold}m")
    if args.clip_to_input_max:
        print(f"[INFO] Clipping model/inpaint outputs to max of tr_elev.npy")
    
    print("\n[INFO] Processing ground truth shards and matching predictions...")
    for sample in tqdm(dataset):
        key = sample.get("__key__")
        if key is None: continue
            
        try:
            gt_elev = decode_npy(sample["gt_elev.npy"])
            gt_sem_onehot = decode_npy(sample["gt_onehot14.npy"])
            
            gt_valid_mask = np.isfinite(gt_elev)
            # We still set invalid GT pixels to 0, as they are masked out anyway
            gt_elev_proc = np.where(gt_valid_mask, gt_elev, 0.0) 
            gt_sem_labels = np.argmax(gt_sem_onehot, axis=-1)
            
            gt_stats = get_elevation_stats(gt_elev_proc, gt_valid_mask)

            tr_elev = decode_npy(sample["tr_elev.npy"])
            tr_inpaint = decode_npy(sample["tr_inpaint.npy"])
            tr_sem_onehot = decode_npy(sample["tr_onehot14.npy"])
            
            max_tr_val = np.inf
            if args.clip_to_input_max:
                tr_finite_mask = np.isfinite(tr_elev)
                if np.any(tr_finite_mask):
                    max_tr_val = np.max(tr_elev[tr_finite_mask])
                else:
                    print(f"Warning: tr_elev for {key} has no finite values. No clipping possible.")

            # --- CHANGE IS HERE ---
            tr_elev_proc = np.nan_to_num(tr_elev, nan=invalid_elev_val, posinf=invalid_elev_val, neginf=invalid_elev_val)
            tr_inpaint_proc = np.nan_to_num(tr_inpaint, nan=invalid_elev_val, posinf=invalid_elev_val, neginf=invalid_elev_val)
            # --- END CHANGE ---
            
            tr_sem_labels = np.argmax(tr_sem_onehot, axis=-1)

            # --- Baseline: input_raw ---
            elev_metrics_raw = compute_elevation_metrics(tr_elev_proc, gt_elev_proc, gt_valid_mask) 
            sem_miou_raw = compute_miou(tr_sem_labels, gt_sem_labels, num_classes, sem_ignore_label)
            sem_bf_raw = compute_bf_score(tr_sem_labels, gt_sem_labels, args.bf_tol, sem_ignore_label)
            raw_stats = get_elevation_stats(tr_elev_proc, gt_valid_mask)
            all_results.append(build_result_row(key, "input_raw", elev_metrics_raw, sem_miou_raw, sem_bf_raw, num_classes, gt_stats, raw_stats))

            # --- Baseline: input_inpaint ---
            tr_inpaint_eval = np.clip(tr_inpaint_proc, a_min=None, a_max=max_tr_val)
            elev_metrics_inpaint = compute_elevation_metrics(tr_inpaint_eval, gt_elev_proc, gt_valid_mask)
            inpaint_stats = get_elevation_stats(tr_inpaint_eval, gt_valid_mask)
            all_results.append(build_result_row(key, "input_inpaint", elev_metrics_inpaint, sem_miou_raw, sem_bf_raw, num_classes, gt_stats, inpaint_stats))

            # --- Models ---
            for model_name in model_names:
                if key not in pred_map[model_name]:
                    continue 
                
                pred_files = pred_map[model_name][key]
                
                if 'elev_path' not in pred_files or 'sem_path' not in pred_files:
                    print(f"Warning: Missing elev or sem file for {key} in {model_name}. Skipping.")
                    continue
                    
                pred_elev = np.load(pred_files['elev_path'])
                
                # --- CHANGE IS HERE ---
                pred_elev_proc = np.nan_to_num(pred_elev, nan=invalid_elev_val, posinf=invalid_elev_val, neginf=invalid_elev_val)
                # --- END CHANGE ---

                with open(pred_files['sem_path'], 'rb') as f:
                    pred_sem_bytes = f.read()
                pred_sem_rgb = decode_png_rgb(pred_sem_bytes)
                
                pred_sem_labels = color12_to_labels(pred_sem_rgb).astype(np.int64)

                pred_elev_eval = np.clip(pred_elev_proc, a_min=None, a_max=max_tr_val)
                elev_metrics_model = compute_elevation_metrics(pred_elev_eval, gt_elev_proc, gt_valid_mask)
                
                if elev_metrics_model['mae'] > args.elevation_mae_threshold:
                    print(f"Warning: Skipping frame {key} for model {model_name}. MAE ({elev_metrics_model['mae']:.2f}) > threshold ({args.elevation_mae_threshold})")
                    continue
                
                sem_miou_model = compute_miou(pred_sem_labels, gt_sem_labels, num_classes, sem_ignore_label)
                sem_bf_model = compute_bf_score(pred_sem_labels, gt_sem_labels, args.bf_tol, sem_ignore_label)
                
                pred_stats = get_elevation_stats(pred_elev_eval, gt_valid_mask)
                
                all_results.append(build_result_row(key, model_name, elev_metrics_model, sem_miou_model, sem_bf_model, num_classes, gt_stats, pred_stats))
        
        except Exception as e:
            print(f"Warning: Failed to process sample {key}. Error: {e}")
            continue

    if not all_results:
        print("No results were processed. Exiting.")
        return

    print("\n[INFO] Aggregating and Saving Results...")
    
    df = pd.DataFrame(all_results)
    csv_per_frame_path = os.path.join(args.output_dir, 'all_models_metrics_per_frame.csv')
    df.to_csv(csv_per_frame_path, index=False)
    print(f"Saved per-frame metrics to: {csv_per_frame_path}")

    df_summary = df.drop(columns=['key']).groupby('model_name').mean()
    csv_summary_path = os.path.join(args.output_dir, 'all_models_evaluation_summary.csv')
    df_summary.to_csv(csv_summary_path)
    
    print(f"\nSaved overall summary metrics to: {csv_summary_path}")
    print("-------------------------------------------------")
    print(df_summary[['mae', 'rmse', 'miou', 'bf_score', 'gt_elev_mean', 'pred_elev_mean']])
    print("-------------------------------------------------")
    
    print("\n[INFO] Generating plots...")
    
    try:
        main_metrics = ['mae', 'rmse', 'miou', 'bf_score']
        for metric in main_metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='model_name', y=metric)
            plt.title(f'Model Comparison: {metric.upper()}')
            plt.ylabel(metric.upper())
            plt.xlabel('Model')
            plt.xticks(rotation=15)
            plot_path = os.path.join(args.output_dir, f'plot_boxplot_{metric}.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_path}")

        iou_cols = [f'iou_class_{i}' for i in range(num_classes)]
        df_summary_iou = df_summary[iou_cols].T
        df_summary_iou.index = [f'Class {i}' for i in range(num_classes)]
        
        df_summary_iou.plot(
            kind='bar', 
            figsize=(15, 7), 
            width=0.8
        )
        plt.title('Mean Per-Class IoU by Model')
        plt.ylabel('mIoU')
        plt.xlabel('Semantic Class')
        plt.legend(title='Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plot_path = os.path.join(args.output_dir, f'plot_barchart_per_class_iou.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

    except Exception as e:
        print(f"Warning: Could not generate plots. Error: {e}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against ground truth test set."
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
        help="Base directory containing all model run folders (e.g., 'all357_unet')."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./model_evaluation",
        help="Directory to save CSVs and plots."
    )
    parser.add_argument(
        "--bf_tol", 
        type=int, 
        default=2, 
        help="Boundary F1 tolerance (pixels), (default: 2)."
    )
    parser.add_argument(
        "--elevation_mae_threshold", 
        type=float, 
        default=100.0, 
        help="Elevation MAE threshold (in meters) to filter outlier frames (default: 100.0)."
    )
    parser.add_argument(
        '--clip_to_input_max', 
        action='store_true', 
        default=True,
        help="Clip model/inpaint outputs to the max of tr_elev.npy (default: True)"
    )
    parser.add_argument(
        '--no_clip', 
        action='store_false', 
        dest='clip_to_input_max',
        help="Do NOT clip model/inpaint outputs to the max of tr_elev.npy"
    )
    
    args = parser.parse_args()
    main(args)