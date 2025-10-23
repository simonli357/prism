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
    from color_mapping import color28_to_onehot14, color28_to_new14_indices_and_color, onehot14_to_color, NEW_CLASSES, SEM_CHANNELS
except Exception:
    from .color_mapping import color28_to_onehot14, color28_to_new14_indices_and_color, onehot14_to_color, NEW_CLASSES, SEM_CHANNELS

from tqdm import tqdm
from PIL import Image

def onehot_to_indices_safe(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr is None or arr.ndim != 3:
        return None
    for axis in (-1, 0, 1):
        c = arr.shape[axis]
        if 4 <= c <= 64:
            return np.argmax(arr, axis=axis)
    return None

def rgb_to_indices(rgb: np.ndarray) -> Optional[np.ndarray]:
    if rgb is None:
        return None
    if color28_to_new14_indices_and_color is None:
        return None
    ids, _ = color28_to_new14_indices_and_color(rgb.astype(np.uint8))
    try:
        import torch
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import cupy as cp
        if isinstance(ids, cp.ndarray):
            ids = cp.asnumpy(ids)
    except Exception:
        pass
    return ids

def load_indices_from_png(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return rgb_to_indices(rgb)

def seg_boundary_map(ids: np.ndarray) -> np.ndarray:
    b = np.zeros_like(ids, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            b |= np.roll(np.roll(ids, dy, axis=0), dx, axis=1) != ids
    return b

def dilate_mask(m: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return m
    out = m.copy()
    for _ in range(tol):
        prev = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                out |= np.roll(np.roll(prev, dy, axis=0), dx, axis=1)
    return out

def boundary_f1_counts(pred_ids: np.ndarray, gt_ids: np.ndarray, tol: int, unlabeled_id: int = 0):
    pb = seg_boundary_map(pred_ids)
    gb = seg_boundary_map(gt_ids)

    pred_lab = pred_ids != unlabeled_id
    gt_lab = gt_ids != unlabeled_id
    pb &= pred_lab
    gb &= gt_lab

    gb_d = dilate_mask(gb, tol)
    pb_d = dilate_mask(pb, tol)

    tp_p = int((pb & gb_d).sum())  # predicted boundary pixels matched to GT
    tp_g = int((gb & pb_d).sum())  # GT boundary pixels matched to pred
    return tp_p, int(pb.sum()), tp_g, int(gb.sum())

def confusion_update(conf: np.ndarray, gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore: int = 0):
    valid = (gt != ignore)
    if not np.any(valid):
        return
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    idx = g * num_classes + p
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    conf += counts.reshape((num_classes, num_classes))
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
    if not shards:
        raise SystemExit(f"No .tar shards found under {test_root}.")

    ds = wds.WebDataset(shards, shardshuffle=False)

    from io import BytesIO

    for sample in ds:
        key = sample.get("__key__")
        if key is None:
            continue
        key = os.path.basename(key)

        gt_b = sample.get("gt_elev.npy")
        tr_b = sample.get("tr_elev.npy")
        if gt_b is None or tr_b is None:
            continue

        gt = npy_from_bytes(gt_b)
        tr = npy_from_bytes(tr_b)

        gt_onehot = None
        oh_b = sample.get("gt_onehot14.npy")
        if oh_b is not None:
            gt_onehot = npy_from_bytes(oh_b)
        else:
            rgb_b = sample.get("gt_color14.png") or sample.get("gt_rgb.png")
            if rgb_b is not None and color28_to_onehot14 is not None:
                rgb = np.array(Image.open(BytesIO(rgb_b)).convert("RGB"), dtype=np.uint8)
                gt_onehot = color28_to_onehot14(rgb, dtype=np.float32)
                try:
                    import torch
                    if isinstance(gt_onehot, torch.Tensor):
                        gt_onehot = gt_onehot.detach().cpu().numpy()
                except Exception:
                    pass
                try:
                    import cupy as cp
                    if isinstance(gt_onehot, cp.ndarray):
                        gt_onehot = cp.asnumpy(gt_onehot)
                except Exception:
                    pass

        tr_onehot = None
        troh_b = sample.get("tr_onehot14.npy")
        if troh_b is not None:
            tr_onehot = npy_from_bytes(troh_b)

        tr_rgb = None
        tr_rgb_b = sample.get("tr_rgb.png") or sample.get("tr_color14.png")
        if tr_rgb_b is not None:
            tr_rgb = np.array(Image.open(BytesIO(tr_rgb_b)).convert("RGB"), dtype=np.uint8)

        yield key, gt, tr, gt_onehot, tr_onehot, tr_rgb
      
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

def collect_infer_semantics(run_dir: str) -> Dict[str, str]:
    out_dir = os.path.join(run_dir, "inference_on_test_set")
    if not os.path.isdir(out_dir):
        return {}
    mapping: Dict[str, str] = {}

    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.startswith("run1_") and f.endswith(".pred_color14.png"):
                key = f.replace(".pred_color14.png", "")
                mapping[key] = os.path.join(root, f)

    tars = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".tar")]
    if tars and wds is not None:
        ds = wds.WebDataset(sorted(tars), shardshuffle=False)
        tmpdir = os.path.join(out_dir, "_wds_tmp_sem")
        os.makedirs(tmpdir, exist_ok=True)
        for sample in ds:
            key = sample.get("__key__")
            if key is None:
                continue
            key = os.path.basename(key)
            pred_b = sample.get("pred_color14.png")
            if pred_b is None:
                for name, payload in sample.items():
                    if isinstance(name, str) and name.endswith(".pred_color14.png"):
                        pred_b = payload
                        break
            if pred_b is None:
                continue
            out_path = os.path.join(tmpdir, f"{key}.pred_color14.png")
            if key not in mapping and not os.path.exists(out_path):
                with open(out_path, "wb") as fh:
                    fh.write(pred_b)
                mapping[key] = out_path
    return mapping
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
def plot_semantic_bars(per_model_sem_summary, out_dir, bf_tol: int):
    set_pub_style()
    labels = list(per_model_sem_summary.keys())
    mIoU = [per_model_sem_summary[k]["mIoU"] for k in labels]
    Pix  = [per_model_sem_summary[k]["PixelAcc"] for k in labels]
    BF   = [per_model_sem_summary[k]["BFscore"] for k in labels]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, vals, title in zip(
        axes,
        [mIoU, Pix, BF],
        ["mIoU (↑)", "Pixel Acc (↑)", f"Boundary F1 @ {bf_tol}px (↑)"],
    ):
        x = np.arange(len(labels)); bars = ax.bar(x, vals)
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "semantics_core_metrics.png"))
    plt.close(fig)

def plot_per_class_iou(per_model_sem_perclass: Dict[str, np.ndarray], class_names: List[str], out_dir: str):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(class_names)+3), 4.0))
    xs = np.arange(1, len(class_names))  # skip unlabeled=0
    for name, ious in per_model_sem_perclass.items():
        ax.plot(xs, ious[1:], marker='o', linewidth=1.5, label=name)
    ax.set_xticks(xs, class_names[1:], rotation=20, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU (ignore unlabeled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_iou_by_model.png"))
    plt.close(fig)

def plot_confusion_heatmap(conf: np.ndarray, class_names: List[str], title: str, out_path: str):
    """Row-normalized confusion (GT on rows)."""
    set_pub_style()
    cm = conf.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_norm = np.divide(cm, np.maximum(row_sums, 1), out=np.zeros_like(cm), where=row_sums>0)
    fig, ax = plt.subplots(figsize=(max(6, 0.45*len(class_names)), max(5, 0.45*len(class_names))))
    im = ax.imshow(cm_norm, interpolation="nearest", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("GT")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)), class_names)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("row-norm freq", rotation=90, va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

BASE_ELEV = "baseline_tr_elev"   # uses tr_elev.npy (elevation-only)
BASE_SEM  = "baseline_tr_sem" 
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
    parser.add_argument("--bf-tol", type=int, default=2,
                        help="Boundary F1 tolerance in pixels.")
    args = parser.parse_args()

    C = SEM_CHANNELS
    class_names = NEW_CLASSES

    set_pub_style()
    os.makedirs(args.out, exist_ok=True)

    cdf_bin_edges = np.linspace(0.0, args.cdf_max_error, args.cdf_bins + 1, dtype=np.float32)

    run_dirs = find_run_dirs(args.runs_root, args.models)
    if not run_dirs:
        raise SystemExit("No run directories found. Check --runs-root and --models.")
    print(f"Discovered runs: {list(run_dirs.keys())}")

    per_model_summary = {}          # elevation scalars
    per_model_abs_errors = {}       # (bin_edges, counts)
    per_model_bins = {}             # height-binned MAE
    per_model_scatter = {}          # sampled (gt, pred) points
    per_model_objmae = {}           # scalar
    per_model_sem_summary = {}      # mIoU, PixelAcc, BFscore
    per_model_sem_perclass = {}     # IoU per class

    per_frame_rows = []

    model_entries = list(run_dirs.items())               # [(name, path)]
    model_entries.insert(0, (BASE_SEM,  None))           # semantics-only baseline (train semantics)
    model_entries.insert(0, (BASE_ELEV, None))           # elevation-only baseline (train elevation)

    for model_key, run_dir in tqdm(model_entries, desc="Models"):
        conf = np.zeros((C, C), dtype=np.int64)
        bf_tp_p = bf_pred = bf_tp_g = bf_gt = 0

        if model_key in (BASE_ELEV, BASE_SEM):
            pred_map = {}
            pred_sem_map = {}
        else:
            pred_map = collect_infer_predictions(run_dir)          # elevation preds (.pred_elev.npy)
            pred_sem_map = collect_infer_semantics(run_dir)        # semantic preds (.pred_color14.png)

        cdf_counts = np.zeros(args.cdf_bins, dtype=np.int64)
        scatter_res = ReservoirSampler(cap=args.scatter_cap, seed=357)

        bin_acc_num = defaultdict(float)
        bin_acc_den = defaultdict(int)

        acc = Counter()
        count_frames = 0

        for key, gt, tr, gt_onehot, tr_onehot, tr_rgb in tqdm(
            iter_gt_samples_wds(args.test_root), leave=False, desc=f"{model_key} frames (wds)"
        ):
            if model_key == BASE_ELEV:
                pred = tr
            elif model_key == BASE_SEM:
                pred = None  # semantics-only baseline
            else:
                pred_p = pred_map.get(key)
                pred = safe_load_npy(pred_p) if pred_p is not None else None

            valid = np.isfinite(gt)

            edge_mask = None
            if gt_onehot is not None:
                indices = onehot_to_indices_safe(gt_onehot)
                if indices is not None:
                    edge_mask = compute_boundary_mask(indices, tol=2)

            if pred is not None:
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

                centers, maes = m["bin_centers"], m["bin_mae"]
                for ccenter, v in zip(centers, maes):
                    bin_acc_num[ccenter] += v
                    bin_acc_den[ccenter] += 1

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
                    **{k: (m[k] if not math.isnan(m[k]) else float("nan"))
                    for k in ["MAE","RMSE","Bias","SIlog","delta1.25","delta1.25^2","EdgeMAE","ObjMAE"]}
                })

            gt_ids = onehot_to_indices_safe(gt_onehot) if gt_onehot is not None else None
            if gt_ids is not None:
                if model_key == BASE_SEM:
                    pred_ids = None
                    if tr_onehot is not None:
                        pred_ids = onehot_to_indices_safe(tr_onehot)
                    if pred_ids is None and tr_rgb is not None and color28_to_onehot14 is not None:
                        tr_oh = color28_to_onehot14(tr_rgb, dtype=np.float32)
                        try:
                            import torch
                            if isinstance(tr_oh, torch.Tensor):
                                tr_oh = tr_oh.detach().cpu().numpy()
                        except Exception:
                            pass
                        try:
                            import cupy as cp
                            if isinstance(tr_oh, cp.ndarray):
                                tr_oh = cp.asnumpy(tr_oh)
                        except Exception:
                            pass
                        pred_ids = onehot_to_indices_safe(tr_oh)
                else:
                    sem_path = pred_sem_map.get(key) if pred_sem_map else None
                    pred_ids = load_indices_from_png(sem_path) if sem_path is not None else None

                if pred_ids is not None and pred_ids.shape == gt_ids.shape:
                    confusion_update(conf, gt_ids, pred_ids, C, ignore=0)
                    tpp, pc, tpg, gc = boundary_f1_counts(pred_ids, gt_ids, tol=args.bf_tol, unlabeled_id=0)
                    bf_tp_p += tpp; bf_pred += pc; bf_tp_g += tpg; bf_gt += gc

            count_frames += 1

        if count_frames == 0:
            print(f"[WARN] No matched frames for {model_key}")
            continue

        if model_key != BASE_ELEV:
            inter = np.diag(conf).astype(np.float64)
            gt_sum = conf.sum(axis=1).astype(np.float64)
            pr_sum = conf.sum(axis=0).astype(np.float64)
            union = gt_sum + pr_sum - inter
            ious = np.full(C, np.nan, dtype=np.float64)
            valid_classes = (union > 0)
            ious[valid_classes] = inter[valid_classes] / union[valid_classes]

            labeled = np.arange(1, C)  # ignore unlabeled=0
            labeled_den = gt_sum[labeled].sum()
            if labeled_den > 0:
                miou = np.nanmean(ious[labeled])
                pix_acc = inter[labeled].sum() / labeled_den
                prec = (bf_tp_p / max(1, bf_pred)) if bf_pred > 0 else 0.0
                rec  = (bf_tp_g / max(1, bf_gt)) if bf_gt > 0 else 0.0
                bf = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0

                per_model_sem_summary[model_key] = {
                    "mIoU": float(miou),
                    "PixelAcc": float(pix_acc),
                    "BFscore": float(bf),
                }
                per_model_sem_perclass[model_key] = ious

                heat_path = os.path.join(args.out, f"confusion_{model_key}.png")
                plot_confusion_heatmap(conf, class_names, f"Confusion (GT rows) — {model_key}", heat_path)
            else:
                print(f"[WARN] No labeled semantics accumulated for {model_key}; skipping mIoU/PixelAcc/BF")

        inter = np.diag(conf).astype(np.float64)
        gt_sum = conf.sum(axis=1).astype(np.float64)
        pr_sum = conf.sum(axis=0).astype(np.float64)
        union = gt_sum + pr_sum - inter
        ious = np.full(C, np.nan, dtype=np.float64)
        valid_classes = (union > 0)
        ious[valid_classes] = inter[valid_classes] / union[valid_classes]

        labeled = np.arange(1, C)  # ignore unlabeled=0
        miou = np.nanmean(ious[labeled])
        pix_acc = inter[labeled].sum() / max(1.0, gt_sum[labeled].sum())
        prec = (bf_tp_p / max(1, bf_pred)) if bf_pred > 0 else 0.0
        rec  = (bf_tp_g / max(1, bf_gt)) if bf_gt > 0 else 0.0
        bf = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0

        per_model_sem_summary[model_key] = {
            "mIoU": float(miou),
            "PixelAcc": float(pix_acc),
            "BFscore": float(bf),
        }
        per_model_sem_perclass[model_key] = ious

        heat_path = os.path.join(args.out, f"confusion_{model_key}.png")
        plot_confusion_heatmap(conf, class_names, f"Confusion (GT rows) — {model_key}", heat_path)

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

    sem_sum_path = os.path.join(args.out, "semantics_summary_per_model.csv")
    with open(sem_sum_path, "w") as f:
        f.write("model,mIoU,PixelAcc,BFscore\n")
        for model_key, s in per_model_sem_summary.items():
            f.write(f"{model_key},{s['mIoU']:.6f},{s['PixelAcc']:.6f},{s['BFscore']:.6f}\n")
    print(f"[OK] Wrote {sem_sum_path}")

    perclass_path = os.path.join(args.out, "semantics_per_class_iou.csv")
    with open(perclass_path, "w") as f:
        f.write("model," + ",".join(class_names) + "\n")
        for model_key, ious in per_model_sem_perclass.items():
            row = [model_key] + [f"{(ious[i] if not np.isnan(ious[i]) else float('nan')):.6f}" for i in range(C)]
            f.write(",".join(row) + "\n")
    print(f"[OK] Wrote {perclass_path}")

    print("[INFO] Generating plots...")
    set_pub_style()

    if per_model_summary:
        print("[INFO] - Elevation bar plots...")
        plot_bars(per_model_summary, args.out)

        print("[INFO] - Elevation CDF plots...")
        plot_cdf_abs_error(per_model_abs_errors, args.out)

        print("[INFO] - Error vs Height plots...")
        plot_error_vs_height_bins(per_model_bins, args.height_bins, args.out)

        print("[INFO] - Object MAE plots...")
        plot_object_mae(per_model_objmae, args.out)

    if per_model_sem_summary:
        plot_semantic_bars(per_model_sem_summary, args.out, args.bf_tol)
        plot_per_class_iou(per_model_sem_perclass, class_names, args.out)
        print(f"[OK] Semantics plots saved to {args.out}")

    print(f"[OK] Plots saved to {args.out}")

if __name__ == "__main__":
    main()
