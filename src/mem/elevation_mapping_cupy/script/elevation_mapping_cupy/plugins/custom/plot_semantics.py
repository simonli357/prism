#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
  python3 plot_semantics.py \
    --runs-root /media/slsecret/T7/carla3/runs \
    --test-root /media/slsecret/T7/carla3/data_split357/test \
    --out /media/slsecret/T7/carla3/analysis_semantics_only \
    --models unet unet_attention deeplabv3p cnn \
    --sem-channels 12 \
    --bf-tol 2
"""

import os, io, glob, json, argparse, csv
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from color_mapping import (
        color28_to_new14_indices_and_color,
        color28_to_onehot14,
    )
except Exception:
    color28_to_new14_indices_and_color = None
    color28_to_onehot14 = None

try:
    import webdataset as wds
except Exception:
    wds = None

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
        "figure.figsize": (7.6, 4.6),
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def find_run_dirs(runs_root: str, models: List[str]) -> Dict[str, str]:
    """Return {run_name: abs_dir} for folders matching '*_<model>'."""
    alias = {"unet_attn": "unet_attention"}
    runs = {}
    for m in models:
        mm = alias.get(m, m)
        pattern = os.path.join(runs_root, f"*_{mm}")
        for d in sorted(glob.glob(pattern)):
            if os.path.isdir(d):
                runs[os.path.basename(d)] = d
    return runs

def npy_from_bytes(b: bytes) -> np.ndarray:
    arr = np.load(io.BytesIO(b), allow_pickle=False)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
    return arr

def onehot_to_indices_safe(arr: np.ndarray) -> Optional[np.ndarray]:
    """Return (H,W) ids from one-hot shaped (H,W,C) or (C,H,W)."""
    if arr is None or arr.ndim != 3:
        return None
    for axis in (-1, 0, 1):
        c = arr.shape[axis]
        if 4 <= c <= 64:
            return np.argmax(arr, axis=axis)
    return None

def rgb_to_indices(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Convert ORIGINAL palette semantic RGB to NEW 12-class indices via color_mapping."""
    if rgb is None:
        return None
    if color28_to_new14_indices_and_color is None:
        raise RuntimeError(
            "RGB→IDs conversion requires color_mapping.py (color28_to_new14_indices_and_color). "
            "Please ensure it is importable."
        )
    ids, _ = color28_to_new14_indices_and_color(rgb.astype(np.uint8))
    # torch / cupy compatibility
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
        print("Note: cupy is not installed; proceeding without it.")
        exit(1)
    return ids

def load_indices_from_png(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return rgb_to_indices(rgb)

# -----------------------------
# Boundary F1 (BF-score) utils
# -----------------------------
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

# -----------------------------
# WebDataset streaming (GT + baseline)
# -----------------------------
def iter_semantic_samples_wds(test_root: str, prefer_onehot=True) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    """
    Yields (key, gt_ids, tr_ids) for every sample in shards under test_root.
    - gt_ids: (H,W) class indices (ignore unlabeled=0)
    - tr_ids: (H,W) baseline (training semantics) as class indices or None
    """
    if wds is None:
        raise RuntimeError("webdataset is not installed. Please `pip install webdataset`.")
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

        # --- GT semantics to IDs ---
        gt_ids = None
        if prefer_onehot and sample.get("gt_onehot14.npy") is not None:
            gt_oh = npy_from_bytes(sample["gt_onehot14.npy"])
            gt_ids = onehot_to_indices_safe(gt_oh)
        if gt_ids is None:
            rgb_b = sample.get("gt_color14.png") or sample.get("gt_rgb.png")
            if rgb_b is not None:
                rgb = np.array(Image.open(BytesIO(rgb_b)).convert("RGB"), dtype=np.uint8)
                gt_ids = rgb_to_indices(rgb)
        if gt_ids is None:
            continue  # cannot evaluate this sample

        # --- TRAIN semantics to IDs (baseline) ---
        tr_ids = None
        if sample.get("tr_onehot14.npy") is not None:
            tr_oh = npy_from_bytes(sample["tr_onehot14.npy"])
            tr_ids = onehot_to_indices_safe(tr_oh)
        if tr_ids is None:
            tr_rgb_b = sample.get("tr_color14.png") or sample.get("tr_rgb.png")
            if tr_rgb_b is not None:
                tr_rgb = np.array(Image.open(BytesIO(tr_rgb_b)).convert("RGB"), dtype=np.uint8)
                tr_ids = rgb_to_indices(tr_rgb)

        yield key, gt_ids, tr_ids

# -----------------------------
# Pred semantics discovery
# -----------------------------
def collect_infer_semantics(run_dir: str) -> Dict[str, str]:
    """
    Return { key: pred_color_png_path } for a run's inference outputs.
    Recurses into subdirs and also scans any *.tar in inference_on_test_set.
    Writes tar members to _wds_tmp_sem/ to reuse.
    """
    out_dir = os.path.join(run_dir, "inference_on_test_set")
    if not os.path.isdir(out_dir):
        return {}
    mapping: Dict[str, str] = {}

    # 1) Extracted files on disk
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.startswith("run1_") and f.endswith(".pred_color14.png"):
                key = f.replace(".pred_color14.png", "")
                mapping[key] = os.path.join(root, f)

    # 2) Inside tars (optional)
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
                # fallback: find any component ending with .pred_color14.png
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

def extract_val_curves(run_dir: str) -> Optional[Tuple[Dict[str, List[Tuple[int, float]]], str]]:
    """
    Return (curves, source) where curves is:
      {'miou': [(epoch, val), ...], 'pix_acc': [...], 'bf_score': [...]}
    Search order (first hit wins):
      - manifest*.json
      - manifest*  (no extension)
      - *.json     (fallback)
      - training_log*.csv
    If nothing usable is found, return None.
    """
    def parse_manifest_json(path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            return None
        vme = data.get("val_metrics_per_epoch")
        if isinstance(vme, list) and len(vme) > 0:
            miou = []
            pix  = []
            bf   = []
            for row in vme:
                try:
                    e = int(row.get("epoch"))
                except Exception:
                    # fall back to 1..N
                    e = len(miou) + 1
                def getf(k):
                    v = row.get(k)
                    return float(v) if v is not None else np.nan
                m = getf("miou")
                p = getf("pix_acc")
                b = getf("bf_score")
                if not np.isnan(m): miou.append((e, m))
                if not np.isnan(p): pix.append((e, p))
                if not np.isnan(b): bf.append((e, b))
            if miou or pix or bf:
                return {"miou": miou, "pix_acc": pix, "bf_score": bf}
        return None

    def parse_training_csv(path: str):
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                miou = []; pix = []; bf = []
                for row in reader:
                    try:
                        epoch = int(row.get("epoch", len(miou)+1))
                    except Exception:
                        epoch = len(miou)+1
                    def parsef(keys):
                        for k in keys:
                            if k in row and row[k] not in (None, "", "nan"):
                                try: return float(row[k])
                                except Exception: pass
                        return np.nan
                    m = parsef(["miou","val_miou","val/miou"])
                    p = parsef(["pix_acc","val_pix_acc","val/pix_acc"])
                    b = parsef(["bf_score","val_bf_score","val/bf_score"])
                    if not np.isnan(m): miou.append((epoch, m))
                    if not np.isnan(p): pix.append((epoch, p))
                    if not np.isnan(b): bf.append((epoch, b))
                if miou or pix or bf:
                    return {"miou": miou, "pix_acc": pix, "bf_score": bf}
        except Exception:
            return None
        return None

    searched = []

    # 1) manifest*.json
    for mp in sorted(glob.glob(os.path.join(run_dir, "manifest*.json"))):
        searched.append(mp)
        curves = parse_manifest_json(mp)
        if curves: return curves, mp

    # 2) manifest* (no extension)
    for mp in sorted(glob.glob(os.path.join(run_dir, "manifest*"))):
        if mp.endswith(".json"):  # already handled
            continue
        searched.append(mp)
        curves = parse_manifest_json(mp)
        if curves: return curves, mp

    # 3) any *.json (fallback, some trainers use other names)
    for mp in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if os.path.basename(mp).startswith("manifest"):
            continue  # already considered
        searched.append(mp)
        curves = parse_manifest_json(mp)
        if curves: return curves, mp

    # 4) training_log*.csv
    for cp in sorted(glob.glob(os.path.join(run_dir, "training_log*.csv"))):
        searched.append(cp)
        curves = parse_training_csv(cp)
        if curves: return curves, cp

    return None

# -----------------------------
# Plotting
# -----------------------------
def plot_semantic_bars(per_model_sem_summary: Dict[str, Dict[str, float]], out_dir: str, bf_tol: int):
    set_pub_style()
    labels = list(per_model_sem_summary.keys())
    mIoU = [per_model_sem_summary[k]["mIoU"] for k in labels]
    Pix  = [per_model_sem_summary[k]["PixelAcc"] for k in labels]
    BF   = [per_model_sem_summary[k]["BFscore"] for k in labels]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, vals, title in zip(
        axes, [mIoU, Pix, BF],
        ["mIoU (↑)", "Pixel Acc (↑)", f"Boundary F1 @ {bf_tol}px (↑)"]
    ):
        x = np.arange(len(labels))
        bars = ax.bar(x, vals)
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
    fig, ax = plt.subplots(figsize=(max(9, 0.65*len(class_names)+3), 4.2))
    xs = np.arange(1, len(class_names))  # skip unlabeled=0
    for name, ious in per_model_sem_perclass.items():
        if ious is None: continue
        ax.plot(xs, ious[1:], marker='o', linewidth=1.6, label=name)
    ax.set_xticks(xs, class_names[1:], rotation=22, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU (ignore unlabeled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_iou_by_model.png"))
    plt.close(fig)

def plot_confusion_heatmap(conf: np.ndarray, class_names: List[str], title: str, out_path: str):
    set_pub_style()
    cm = conf.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_norm = np.divide(cm, np.maximum(row_sums, 1), out=np.zeros_like(cm), where=row_sums>0)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5*len(class_names)), max(5.5, 0.5*len(class_names))))
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

def plot_val_curves(per_model_curves: Dict[str, Dict[str, List[Tuple[int, float]]]], out_dir: str):
    set_pub_style()
    # One figure per metric
    metrics = [("miou","mIoU (val) ↑"), ("pix_acc","Pixel Acc (val) ↑"), ("bf_score","Boundary F1 (val) ↑")]
    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        for model, curves in per_model_curves.items():
            if key not in curves or not curves[key]:
                continue
            xs, ys = zip(*sorted(curves[key], key=lambda t: t[0]))
            ax.plot(xs, ys, marker='o', linewidth=1.6, label=model)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{key}_per_epoch.png"))
        plt.close(fig)

# -----------------------------
# Main
# -----------------------------
BASE_SEM = "baseline_tr_sem"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="+", default=["cnn","unet","unet_attn","deeplabv3p"],
                        help="Model name suffixes to look for (dirs like *_<model>)")
    parser.add_argument("--sem-channels", type=int, default=12, help="Number of classes incl. unlabeled=0.")
    parser.add_argument("--bf-tol", type=int, default=2, help="Boundary F1 tolerance (pixels).")
    parser.add_argument("--class-names", nargs="*", default=None,
                        help="Optional names (len==C). index 0 is 'unlabeled'.")
    args = parser.parse_args()

    set_pub_style()
    os.makedirs(args.out, exist_ok=True)

    C = args.sem_channels
    if args.class_names is None or len(args.class_names) != C:
        class_names = [f"C{i}" for i in range(C)]
        class_names[0] = "unlabeled"
    else:
        class_names = args.class_names

    # Discover run dirs
    run_dirs = find_run_dirs(args.runs_root, args.models)
    if not run_dirs:
        raise SystemExit("No run directories found. Check --runs-root and --models.")
    print(f"Discovered runs: {list(run_dirs.keys())}")

    # Build model list incl. baseline
    model_entries = list(run_dirs.items())
    model_entries.insert(0, (BASE_SEM, None))  # baseline uses training semantics in test shards

    # Prepare per-model accumulators
    per_model_sem_summary: Dict[str, Dict[str, float]] = {}
    per_model_sem_perclass: Dict[str, np.ndarray] = {}
    per_model_confusions: Dict[str, np.ndarray] = {}

    # For per-epoch curves
    per_model_curves: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    # Pre-load predicted semantics maps for real models
    pred_maps: Dict[str, Dict[str, str]] = {}
    for name, run_dir in model_entries:
        if name == BASE_SEM:
            continue
        pred_maps[name] = collect_infer_semantics(run_dir)

        # Extract per-epoch curves if available
        curves_res = extract_val_curves(run_dir)
        if curves_res:
            curves_dict, src = curves_res            # <-- unpack
            per_model_curves[name] = curves_dict     # <-- store only the dict
            print(f"[curves] {name}: {src}")

    # Stream test shards once and accumulate per model
    # (We must pass through the dataset for each model anyway to open their predictions,
    #  but we keep the conversion logic centralized.)
    for model_key, run_dir in tqdm(model_entries, desc="Models"):
        conf = np.zeros((C, C), dtype=np.int64)
        bf_tp_p = bf_pred = bf_tp_g = bf_gt = 0

        mapping = {} if model_key == BASE_SEM else pred_maps.get(model_key, {})

        for key, gt_ids, tr_ids in tqdm(iter_semantic_samples_wds(args.test_root), leave=False, desc=f"{model_key} frames (wds)"):
            # predicted ids
            if model_key == BASE_SEM:
                pred_ids = tr_ids
            else:
                sem_path = mapping.get(key)
                pred_ids = load_indices_from_png(sem_path) if sem_path else None

            if pred_ids is None or pred_ids.shape != gt_ids.shape:
                continue

            # accumulate confusion and BF counts
            confusion_update(conf, gt_ids, pred_ids, C, ignore=0)
            tpp, pc, tpg, gc = boundary_f1_counts(pred_ids, gt_ids, tol=args.bf_tol, unlabeled_id=0)
            bf_tp_p += tpp; bf_pred += pc; bf_tp_g += tpg; bf_gt += gc

        # finalize semantics for this model
        inter = np.diag(conf).astype(np.float64)
        gt_sum = conf.sum(axis=1).astype(np.float64)
        pr_sum = conf.sum(axis=0).astype(np.float64)
        union = gt_sum + pr_sum - inter
        ious = np.full(C, np.nan, dtype=np.float64)
        valid = (union > 0)
        ious[valid] = inter[valid] / union[valid]

        labeled = np.arange(1, C)
        labeled_den = gt_sum[labeled].sum()
        if labeled_den > 0:
            miou = float(np.nanmean(ious[labeled]))
            pix_acc = float(inter[labeled].sum() / labeled_den)
            prec = (bf_tp_p / max(1, bf_pred)) if bf_pred > 0 else 0.0
            rec  = (bf_tp_g / max(1, bf_gt)) if bf_gt > 0 else 0.0
            bf = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
        else:
            miou = pix_acc = bf = float("nan")

        per_model_sem_summary[model_key] = {"mIoU": miou, "PixelAcc": pix_acc, "BFscore": bf}
        per_model_sem_perclass[model_key] = ious
        per_model_confusions[model_key] = conf

        # confusion heatmap
        heat_path = os.path.join(args.out, f"confusion_{model_key}.png")
        plot_confusion_heatmap(conf, class_names, f"Confusion (GT rows) — {model_key}", heat_path)

    # Write CSVs
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

    # Per-epoch curves CSV (if available)
    if per_model_curves:
        curves_csv = os.path.join(args.out, "semantics_val_per_epoch.csv")
        with open(curves_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model","epoch","miou","pix_acc","bf_score"])
            for model, d in per_model_curves.items():
                # unify epochs present in any of the lists
                epochs = sorted(set([e for e,_ in d.get("miou", [])] + [e for e,_ in d.get("pix_acc", [])] + [e for e,_ in d.get("bf_score", [])]))
                md = { "miou": dict(d.get("miou", [])),
                       "pix_acc": dict(d.get("pix_acc", [])),
                       "bf_score": dict(d.get("bf_score", [])) }
                for e in epochs:
                    writer.writerow([model, e,
                                     f"{md['miou'].get(e, np.nan):.6f}",
                                     f"{md['pix_acc'].get(e, np.nan):.6f}",
                                     f"{md['bf_score'].get(e, np.nan):.6f}"])
        print(f"[OK] Wrote {curves_csv}")

    # Plots
    plot_semantic_bars(per_model_sem_summary, args.out, args.bf_tol)
    plot_per_class_iou(per_model_sem_perclass, class_names, args.out)

    if per_model_curves:
        plot_val_curves(per_model_curves, args.out)

    print(f"[OK] Plots saved to {args.out}")

if __name__ == "__main__":
    main()
