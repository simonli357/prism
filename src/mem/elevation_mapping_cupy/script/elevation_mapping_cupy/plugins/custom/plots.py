#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
  python3 plots.py \
    --runs-root /media/slsecret/T7/carla3/runs \
    --test-root /media/slsecret/T7/carla3/data_split357/test \
    --out /media/slsecret/T7/carla3/analysis_all357 \
    --models unet unet_attention deeplabv3p cnn \
    --height-bins 0 0.5 1 2 4 8 16
"""

import os, re, json, argparse, glob, math, random, itertools, glob
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import webdataset as wds
import io

try:
    from color_mapping import color28_to_onehot14  
except Exception:
    from .color_mapping import color28_to_onehot14  

from tqdm import tqdm
from PIL import Image

class ReservoirSampler:
    def __init__(self, cap: int, seed: int = 357):
        self.cap = int(cap)
        self.rng = np.random.RandomState(seed)
        self.n = 0
        self.gt = None
        self.pr = None

    def add_batch(self, gt: np.ndarray, pr: np.ndarray):
        assert gt.shape == pr.shape
        m = gt.size
        if m == 0 or self.cap <= 0:
            return
        flat_gt = gt.ravel()
        flat_pr = pr.ravel()
        if self.n < self.cap:
            take = min(self.cap - self.n, m)
            if self.gt is None:
                self.gt = flat_gt[:take].copy()
                self.pr = flat_pr[:take].copy()
            else:
                self.gt = np.concatenate([self.gt, flat_gt[:take]])
                self.pr = np.concatenate([self.pr, flat_pr[:take]])
            self.n += take
        for i in range(self.n, self.n + (m - max(0, self.cap - self.n))):
            j = self.rng.randint(0, i + 1)
            src_idx = i - self.n + max(0, self.cap - self.n)
            if j < self.cap:
                self.gt[j] = flat_gt[src_idx]
                self.pr[j] = flat_pr[src_idx]
        self.n += (m - max(0, self.cap - self.n))

    def arrays(self):
        if self.gt is None:
            return np.array([]), np.array([])
        return self.gt, self.pr
def load_onehot_from_rgb(rgb_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(rgb_path) or color28_to_onehot14 is None:
        return None
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    oh = color28_to_onehot14(rgb, dtype=np.float32)
    try:
        import torch
        if isinstance(oh, torch.Tensor):
            oh = oh.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import cupy as cp
        if isinstance(oh, cp.ndarray):
            oh = cp.asnumpy(oh)
    except Exception:
        pass
    return oh
def onehot_to_indices_safe(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr.ndim != 3:
        return None
    for axis in (-1, 0, 1):
        cdim = arr.shape[axis]
        if 4 <= cdim <= 64:
            return np.argmax(arr, axis=axis)
    return None
def npy_from_bytes(b: bytes) -> np.ndarray:
    bio = io.BytesIO(b)
    arr = np.load(bio, allow_pickle=False)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
def iter_gt_samples_wds(test_root: str):
    shards = sorted(glob.glob(os.path.join(test_root, "*.tar")))
    ds = wds.WebDataset(shards, shardshuffle=False)

    for sample in ds:
        key = sample.get("__key__")
        if key is None:
            continue
        key = os.path.basename(key)
        gt_b = sample.get("gt_elev.npy")
        tr_b = sample.get("tr_elev.npy")
        if key is None or gt_b is None or tr_b is None:
            continue
        gt = npy_from_bytes(gt_b)
        tr = npy_from_bytes(tr_b)
        oh_b = sample.get("gt_onehot14.npy")
        if oh_b is not None:
            gt_onehot = npy_from_bytes(oh_b)
        else:
            rgb_b = sample.get("gt_color14.png") or sample.get("gt_rgb.png")
            if rgb_b is not None and color28_to_onehot14 is not None:
                from io import BytesIO
                rgb = np.array(Image.open(BytesIO(rgb_b)).convert("RGB"), dtype=np.uint8)
                gt_onehot = load_onehot_from_rgb(Path("dummy").as_posix())  # reuse converter
                gt_onehot = color28_to_onehot14(rgb, dtype=np.float32)
                try:
                    import torch
                    if isinstance(gt_onehot, torch.Tensor):
                        gt_onehot = gt_onehot.detach().cpu().numpy()
                except Exception:
                    pass
            else:
                gt_onehot = None
        yield key, gt, tr, gt_onehot
        
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.figsize": (7.2, 4.6),
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# -----------------------------
# IO helpers
# -----------------------------
def find_run_dirs(runs_root: str, models: List[str]) -> Dict[str, str]:
    """
    Return {model_name: run_dir} mapping for the latest/all seeds it finds per model.
    We pick all matching dirs; key is e.g. 'all357_unet' for uniqueness.
    """
    runs = {}
    for m in models:
        pattern = os.path.join(runs_root, f"*_{m}")
        for d in sorted(glob.glob(pattern)):
            if os.path.isdir(d):
                runs[os.path.basename(d)] = d
    return runs  # keys like 'all357_unet', values are absolute paths

def collect_test_index(test_root: str) -> Dict[str, Dict[str, str]]:
    """
    Build an index from frame key (e.g., 'run1_000000123') to file paths in GT shards.
    Returns: { key: {"gt_elev": path, "tr_elev": path, "gt_onehot": path} }
    """
    index = {}
    shard_dirs = []
    for root, dirs, files in os.walk(test_root):
        # only leaf dirs that contain run1_*.npy files
        if any(f.startswith("run1_") for f in files):
            shard_dirs.append(root)

    for d in shard_dirs:
        for f in os.listdir(d):
            if f.startswith("run1_") and f.endswith(".npy"):
                stem = f.split(".")[0]  # run1_xxxxxxxx.<rest>
                key = stem  # use full 'run1_000000123'
                rec = index.setdefault(key, {})
                if f.endswith("gt_elev.npy"):
                    rec["gt_elev"] = os.path.join(d, f)
                elif f.endswith("tr_elev.npy"):
                    rec["tr_elev"] = os.path.join(d, f)
                elif f.endswith("gt_onehot14.npy"):
                    rec["gt_onehot"] = os.path.join(d, f)
    return index

def collect_infer_predictions(run_dir: str) -> Dict[str, str]:
    out_dir = os.path.join(run_dir, "inference_on_test_set")
    if not os.path.isdir(out_dir):
        return {}
    mapping: Dict[str, str] = {}

    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.startswith("run1_") and f.endswith(".pred_elev.npy"):
                key = f[:-len(".pred_elev.npy")]
                mapping[key] = os.path.join(root, f)

    tars = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".tar")]
    if tars and wds is not None:
        ds = wds.WebDataset(sorted(tars), shardshuffle=False)
        tmpdir = os.path.join(out_dir, "_wds_tmp")
        os.makedirs(tmpdir, exist_ok=True)

        for sample in ds:
            key = sample.get("__key__")
            if key is None:
                continue
            key = os.path.basename(key)  # ensure it's like run1_000000301

            pred_b = sample.get("pred_elev.npy")

            if pred_b is None:
                for comp_name, comp_bytes in sample.items():
                    if isinstance(comp_name, str) and comp_name.endswith(".pred_elev.npy"):
                        pred_b = comp_bytes
                        break

            if pred_b is None:
                continue  

            out_path = os.path.join(tmpdir, f"{key}.pred_elev.npy")
            if key not in mapping and not os.path.exists(out_path):
                with open(out_path, "wb") as fh:
                    fh.write(pred_b)
                mapping[key] = out_path

    return mapping

def safe_load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def compute_boundary_mask(gt_indices: np.ndarray, tol: int = 2) -> np.ndarray:
    """
    Build a boolean mask (H,W) where pixel is within tol px of a semantic boundary.
    """
    h, w = gt_indices.shape
    # boundary pixels: any neighbor different
    b = np.zeros_like(gt_indices, dtype=bool)
    # 4/8-neighborhood diff
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(gt_indices, dy, axis=0), dx, axis=1)
            b |= (shifted != gt_indices)
    # dilate by tol using rolling (cheap approximation)
    bb = b.copy()
    for _ in range(tol):
        bb_pad = bb.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                bb |= np.roll(np.roll(bb_pad, dy, axis=0), dx, axis=1)
    return bb

def si_log_rmse(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, eps=1e-6) -> float:
    pv = np.clip(pred[valid], eps, None)
    gv = np.clip(gt[valid], eps, None)
    d = np.log(pv) - np.log(gv)
    return float(np.sqrt(np.mean(d**2) - (np.mean(d)**2)))

def delta_accuracy(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, thresh: float, eps=1e-6) -> float:
    pv = np.clip(pred[valid], eps, None)
    gv = np.clip(gt[valid], eps, None)
    ratio = np.maximum(pv / gv, gv / pv)
    return float(np.mean(ratio < thresh))

def per_class_object_mae(pred: np.ndarray, gt: np.ndarray, gt_onehot: np.ndarray,
                         class_ids: List[int], valid: np.ndarray) -> float:
    if gt_onehot.ndim != 3:
        return float("nan")
    indices = onehot_to_indices_safe(gt_onehot)
    if indices is None:
        return float("nan")
    mask_total = 0
    err_sum = 0.0
    for cid in class_ids:
        m = (indices == cid) & valid
        if np.any(m):
            err_sum += np.abs(pred[m] - gt[m]).sum()
            mask_total += m.sum()
    if mask_total == 0:
        return float("nan")
    return float(err_sum / mask_total)

def height_binned_mae(pred, gt, valid, bins: List[float]):
    """
    Return centers and MAE per height bin based on GT height.
    """
    gv = gt[valid]
    pv = pred[valid]
    bins = np.asarray(bins, dtype=float)
    idx = np.digitize(gv, bins, right=False)
    # bins => intervals [bins[i], bins[i+1))
    maes = []
    centers = []
    for i in range(len(bins) - 1):
        sel = (idx == i+1)
        if np.any(sel):
            e = np.abs(pv[sel] - gv[sel])
            maes.append(float(e.mean()))
            centers.append(0.5 * (bins[i] + bins[i+1]))
    return np.array(centers), np.array(maes)

def compute_all_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                        gt_onehot: Optional[np.ndarray],
                        edge_mask: Optional[np.ndarray],
                        obj_classes: List[int],
                        bins: List[float]):
    diff = (pred - gt)
    abs_diff = np.abs(diff)
    mae = float(abs_diff[valid].mean()) if np.any(valid) else float("nan")
    rmse = float(np.sqrt(np.mean((diff[valid])**2))) if np.any(valid) else float("nan")
    bias = float(diff[valid].mean()) if np.any(valid) else float("nan")
    silog = si_log_rmse(pred, gt, valid)
    d1 = delta_accuracy(pred, gt, valid, 1.25)
    d2 = delta_accuracy(pred, gt, valid, 1.25**2)
    edge_mae = float(np.mean(abs_diff[edge_mask & valid])) if edge_mask is not None and np.any(edge_mask & valid) else float("nan")
    obj_mae = per_class_object_mae(pred, gt, gt_onehot, obj_classes, valid) if gt_onehot is not None else float("nan")
    centers, bin_mae = height_binned_mae(pred, gt, valid, bins)
    return {
        "MAE": mae, "RMSE": rmse, "Bias": bias, "SIlog": silog, "delta1.25": d1, "delta1.25^2": d2,
        "EdgeMAE": edge_mae, "ObjMAE": obj_mae,
        "bin_centers": centers, "bin_mae": bin_mae,
        "abs_errors": abs_diff[valid],  # for CDF
        "gt_vals": gt[valid],           # for scatter
        "pred_vals": pred[valid],
    }

# -----------------------------
# Plotting
# -----------------------------
def bar_with_values(ax, labels, values, title, ylabel, ylim=None):
    x = np.arange(len(labels))
    bars = ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=15, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    for b,v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=10)

def plot_bars(summary, out_dir):
    # summary: {model_key: {metric: value}}
    labels = list(summary.keys())
    metrics = ["MAE", "RMSE", "Bias", "delta1.25"]
    fig, axes = plt.subplots(2, 2, figsize=(9,7))
    axes = axes.ravel()
    for ax, m in zip(axes, metrics):
        vals = [summary[k][m] for k in labels]
        ylim = None
        if m == "delta1.25":
            ylim = (0.0, 1.0)
        bar_with_values(ax, labels, vals, m, m)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bars_core_metrics.png"))
    plt.close(fig)

def plot_cdf_abs_error(per_model_abs_errors, out_dir):
    set_pub_style()
    fig, ax = plt.subplots()
    for name, (bin_edges, counts) in per_model_abs_errors.items():
        total = counts.sum()
        if total == 0:
            continue
        cdf = np.cumsum(counts) / float(total)
        x = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # bin centers
        ax.plot(x, cdf, label=name)
    ax.set_xlabel("|error| (m)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of absolute elevation error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cdf_abs_error.png"))
    plt.close(fig)

def plot_error_vs_height_bins(per_model_bins, bins, out_dir):
    set_pub_style()
    fig, ax = plt.subplots()
    for name, (centers, maes) in per_model_bins.items():
        if centers.size == 0: continue
        ax.plot(centers, maes, marker='o', label=name)
    ax.set_xlabel("GT height (m)")
    ax.set_ylabel("MAE (m)")
    ax.set_title("MAE vs GT Height")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mae_vs_height_bins.png"))
    plt.close(fig)

def plot_object_mae(per_model_objmae, out_dir):
    set_pub_style()
    labels = list(per_model_objmae.keys())
    vals = [per_model_objmae[k] for k in labels]
    fig, ax = plt.subplots()
    bar_with_values(ax, labels, vals, "Object-centric MAE (macro over classes)", "MAE (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "object_mae_macro.png"))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", default=["cnn","unet","unet_attn","deeplabv3p"],
                        help="Model name suffixes to look for (dirs like *_<model>)")
    parser.add_argument("--height-bins", nargs="+", type=float, default=[0,0.5,1,2,4,8,16,32],
                        help="Bin edges for GT height (meters).")
    parser.add_argument("--obj-class-ids", nargs="*", type=int, default=[2,3,4,5,6,8,10,11],
                        help="GT semantic class ids to treat as 'objects' for Obj-MAE.")
    parser.add_argument("--seeded-subsample", type=int, default=10000,
                        help="Max pixels per model to keep for CDF & scatter plots.")
    parser.add_argument("--cdf-max-error", type=float, default=15.0,
                        help="Clamp |error| for CDF to this max (meters).")
    parser.add_argument("--cdf-bins", type=int, default=256,
                        help="Number of histogram bins for CDF (fixed memory).")
    parser.add_argument("--scatter-cap", type=int, default=30000,
                        help="Max total GT/Pred points per model (reservoir sample).")
    args = parser.parse_args()

    set_pub_style()
    os.makedirs(args.out, exist_ok=True)
    
    cdf_bin_edges = np.linspace(0.0, args.cdf_max_error, args.cdf_bins + 1, dtype=np.float32)

    run_dirs = find_run_dirs(args.runs_root, args.models)
    if not run_dirs:
        raise SystemExit("No run directories found. Check --runs-root and --models.")
    print(f"Discovered runs: {list(run_dirs.keys())}")

    test_tars = [f for f in os.listdir(args.test_root) if f.endswith(".tar")]

    per_model_summary = {}
    per_model_abs_errors = {}
    per_model_bins = {}
    per_model_scatter = {}
    per_model_objmae = {}

    per_frame_rows = []

    model_entries = list(run_dirs.items())  # [(name, path)]
    model_entries.insert(0, ("baseline_tr_elev", None))

    for model_key, run_dir in tqdm(model_entries, desc="Models"):
        if model_key == "baseline_tr_elev":
            pred_map = {}  
        else:
            pred_map = collect_infer_predictions(run_dir)
            if not pred_map:
                print(f"[WARN] No predictions found for {model_key} at {run_dir}")

        cdf_counts = np.zeros(args.cdf_bins, dtype=np.int64)
        scatter_res = ReservoirSampler(cap=args.scatter_cap, seed=357)

        bin_acc_num = defaultdict(float)
        bin_acc_den = defaultdict(int)

        obj_mae_list = []

        acc = Counter()
        count_frames = 0

        if model_key == "baseline_tr_elev":
            pred_map = {}  # we'll use tr_elev from GT sample itself
        else:
            pred_map = collect_infer_predictions(run_dir)
        for key, gt, tr, gt_onehot in tqdm(iter_gt_samples_wds(args.test_root), leave=False, desc=f"{model_key} frames (wds)"):
            if model_key == "baseline_tr_elev":
                pred = tr
            else:
                pred_p = pred_map.get(key)
                if pred_p is None:
                    continue
                pred = safe_load_npy(pred_p)

            valid = np.isfinite(gt)

            edge_mask = None
            if gt_onehot is not None:
                indices = onehot_to_indices_safe(gt_onehot)
                if indices is not None:
                    edge_mask = compute_boundary_mask(indices, tol=2)

            m = compute_all_metrics(
                pred, gt, valid,
                gt_onehot=gt_onehot,
                edge_mask=edge_mask,
                obj_classes=args.obj_class_ids,
                bins=args.height_bins
            )

            for k in ["MAE","RMSE","Bias","SIlog","delta1.25","delta1.25^2","EdgeMAE","ObjMAE"]:
                if not math.isnan(m[k]):
                    acc[k] += m[k]
            count_frames += 1

            centers, maes = m["bin_centers"], m["bin_mae"]
            for c, v in zip(centers, maes):
                bin_acc_num[c] += v
                bin_acc_den[c] += 1

            if m["abs_errors"].size:
                errs = np.minimum(m["abs_errors"], args.cdf_max_error)
                cdf_counts += np.histogram(errs, bins=cdf_bin_edges)[0]
                if m["gt_vals"].size > args.seeded_subsample:
                    rs = np.random.RandomState(357)
                    sel = rs.choice(m["gt_vals"].size, size=args.seeded_subsample, replace=False)
                    scatter_res.add_batch(m["gt_vals"][sel], m["pred_vals"][sel])
                else:
                    scatter_res.add_batch(m["gt_vals"], m["pred_vals"])

            per_frame_rows.append({
                "model": model_key,
                "frame": key,
                **{k: m[k] for k in ["MAE","RMSE","Bias","SIlog","delta1.25","delta1.25^2","EdgeMAE","ObjMAE"]}
            })

        if count_frames == 0:
            print(f"[WARN] No matched frames for {model_key}")
            continue

        summary = {k: (acc[k]/count_frames) for k in acc.keys()}
        per_model_summary[model_key] = summary

        per_model_abs_errors[model_key] = (cdf_bin_edges, cdf_counts)
        per_model_scatter[model_key] = scatter_res.arrays()

        centers_sorted = sorted(bin_acc_num.keys())
        c_arr = np.array(centers_sorted)
        m_arr = np.array([bin_acc_num[c]/max(1,bin_acc_den[c]) for c in centers_sorted])
        per_model_bins[model_key] = (c_arr, m_arr)

        per_model_objmae[model_key] = summary.get("ObjMAE", float("nan"))

    os.makedirs(args.out, exist_ok=True)
    sum_path = os.path.join(args.out, "summary_per_model.csv")
    all_metrics = ["MAE","RMSE","Bias","SIlog","delta1.25","delta1.25^2","EdgeMAE","ObjMAE"]
    with open(sum_path, "w") as f:
        f.write("model," + ",".join(all_metrics) + "\n")
        for model_key, s in per_model_summary.items():
            f.write(model_key + "," + ",".join(f"{s.get(m, float('nan')):.6f}" for m in all_metrics) + "\n")
    print(f"[OK] Wrote {sum_path}")

    pf_path = os.path.join(args.out, "per_frame_metrics.csv")
    with open(pf_path, "w") as f:
        headers = ["model","frame"] + all_metrics
        f.write(",".join(headers) + "\n")
        for r in per_frame_rows:
            f.write(",".join([r["model"], r["frame"]] + [f"{r.get(m, float('nan')):.6f}" for m in all_metrics]) + "\n")
    print(f"[OK] Wrote {pf_path}")

    print("[INFO] Generating plots...")
    set_pub_style()

    print("[INFO] - Bar plots...")
    plot_bars(per_model_summary, args.out)

    print("[INFO] - CDF plots...")
    plot_cdf_abs_error(per_model_abs_errors, args.out)

    print("[INFO] - Error vs Height plots...")
    plot_error_vs_height_bins(per_model_bins, args.height_bins, args.out)

    print("[INFO] - Object MAE plots...")
    plot_object_mae(per_model_objmae, args.out)

    print(f"[OK] Plots saved to {args.out}")

if __name__ == "__main__":
    main()
