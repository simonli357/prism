#!/usr/bin/env python3
"""
Remap all WebDataset shards in a directory:
- Replace gt_rgb.png / tr_rgb.png with 14-class color mapping
- Add gt_onehot14.npy / tr_onehot14.npy (H,W,14; uint8 or float32)
- Filter out 'Vehicle' pixels with low elevation (< 0.3)

Usage:
  python3 remap_shards.py \
    --in_dir /media/slsecret/T7/carla3/data/town7/gridmap_wds \
    --out_dir /media/slsecret/T7/carla3/data/town7/gridmap_wds_remapped \
    --recursive true \
    --onehot_dtype uint8 \
    --compression 3 \
    --keep_orig_png false \
    --overwrite false
"""

import os
import io
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2
import webdataset as wds
from tqdm import tqdm

try:
    from .color_mapping import (color28_to_new14_indices_and_color, onehot14_to_color,
                                NEW_CLASSES_VEHICLE_ID, NEW_CLASSES_UNLABELED_ID, SEM_CHANNELS,
                                NEW_CLASSES_PERSON_ID, NEW_CLASSES_TWOWHEELER_ID)
except ImportError:
    from color_mapping import (color28_to_new14_indices_and_color, onehot14_to_color,
                               NEW_CLASSES_VEHICLE_ID, NEW_CLASSES_UNLABELED_ID, SEM_CHANNELS,
                               NEW_CLASSES_PERSON_ID, NEW_CLASSES_TWOWHEELER_ID)
    
def _elev_to_heatmap_png(elev: np.ndarray, compression: int = 3) -> bytes:
    """
    Convert float32 elevation (H,W) to a heatmap PNG.
    NaN/inf -> black. Uses OpenCV JET colormap.
    """
    e = np.asarray(elev, dtype=np.float32)
    mask = np.isfinite(e)

    if not np.any(mask):
        img_bgr = np.zeros((*e.shape, 3), dtype=np.uint8)
    else:
        vmin = float(np.min(e[mask]))
        vmax = float(np.max(e[mask]))
        rng = max(vmax - vmin, 1e-6)
        norm = np.zeros_like(e, dtype=np.float32)
        norm[mask] = (e[mask] - vmin) / rng
        img_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img_bgr[~mask] = 0

    ok, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise RuntimeError("Elevation heatmap PNG encoding failed")
    return buf.tobytes()

def _decode_png_to_rgb(img_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes to RGB uint8 array (handles BGR/BGRA)."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError("cv2.imdecode failed on PNG data")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    elif im.ndim == 3:
        if im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
        else:
            c = im.shape[2]
            im = im[:, :, :3] if c > 3 else np.repeat(im, 3 // max(1, c), axis=2)
    if im.dtype != np.uint8:
        im = im.astype(np.uint8)
    return im

def _encode_rgb_to_png(rgb: np.ndarray, compression: int = 3) -> bytes:
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()

def _npy_bytes(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue()


def _process_sample(sample: Dict[str, Any],
                    onehot_dtype: str = "uint8",
                    compression: int = 3,
                    keep_orig_png: bool = False) -> Dict[str, Any]:
    out = {"__key__": sample["__key__"]}

    for k, v in sample.items():
        if k not in ("__key__", "gt_rgb.png", "tr_rgb.png", "gt_elev.npy", "tr_elev.npy", "tr_inpaint.npy"):
            out[k] = v

    gt_elev = None
    if "gt_elev.npy" in sample:
        try:
            gt_elev = np.load(io.BytesIO(sample["gt_elev.npy"]))
            out["gt_elev.npy"] = sample["gt_elev.npy"]
            out["gt_elev_viz.png"] = _elev_to_heatmap_png(gt_elev, compression=compression)
        except Exception as e:
            print(f"[WARN] Could not load or visualize gt_elev.npy for {sample.get('__key__','?')}: {e}")

    tr_elev = None
    if "tr_elev.npy" in sample:
        tr_elev = np.load(io.BytesIO(sample["tr_elev.npy"]))
        out["tr_elev.npy"] = sample["tr_elev.npy"]

    if "tr_inpaint.npy" in sample:
        try:
            tr_inpaint = np.load(io.BytesIO(sample["tr_inpaint.npy"]))
            out["tr_inpaint.npy"] = sample["tr_inpaint.npy"]
            out["tr_inpaint_viz.png"] = _elev_to_heatmap_png(tr_inpaint, compression=compression)
        except Exception as e:
            print(f"[WARN] Could not load or visualize tr_inpaint.npy for {sample.get('__key__','?')}: {e}")
            
    def handle(name_in: str, base: str, elev_map: Optional[np.ndarray]):
        if name_in not in sample:
            return

        rgb = _decode_png_to_rgb(sample[name_in])
        
        new_ids, _ = color28_to_new14_indices_and_color(rgb)

        if elev_map is not None:
            h, w = new_ids.shape
            eh, ew = elev_map.shape
            if h != eh or w != ew:
                elev_map = elev_map[:h, :w]

            target_classes = [
                NEW_CLASSES_VEHICLE_ID,
                # NEW_CLASSES_PERSON_ID,
                NEW_CLASSES_TWOWHEELER_ID
            ]
            is_target_class_mask = np.isin(new_ids, target_classes)
            
            low_elevation_mask = (np.nan_to_num(elev_map, nan=np.inf) < 0.3)
            
            combined_mask = is_target_class_mask & low_elevation_mask
            
            new_ids[combined_mask] = NEW_CLASSES_UNLABELED_ID
        
        target_dtype = np.uint8 if onehot_dtype == "uint8" else np.float32
        onehot = np.eye(SEM_CHANNELS, dtype=target_dtype)[new_ids]
        
        rgb14_roundtrip = onehot14_to_color(onehot)
        
        out[f"{base}_rgb.png"] = _encode_rgb_to_png(rgb14_roundtrip, compression=compression)
        out[f"{base}_onehot14.npy"] = _npy_bytes(onehot)
        if keep_orig_png:
            out[f"{base}_rgb_orig.png"] = sample[name_in]

    handle("gt_rgb.png", "gt", elev_map=gt_elev)
    handle("tr_rgb.png", "tr", elev_map=None)

    return out

def process_shard(in_tar: Path,
                  out_tar: Path,
                  onehot_dtype: str,
                  compression: int,
                  keep_orig_png: bool,
                  overwrite: bool):
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    if out_tar.exists() and not overwrite:
        print(f"[SKIP] {out_tar} exists (use --overwrite true to replace).")
        return

    ds = wds.WebDataset(str(in_tar), handler=wds.warn_and_continue)
    count = 0
    with wds.TarWriter(str(out_tar)) as sink:
        for sample in tqdm(ds, desc=f"Processing {in_tar.name}", unit="sample"):
            try:
                new_sample = _process_sample(
                    sample,
                    onehot_dtype=onehot_dtype,
                    compression=compression,
                    keep_orig_png=keep_orig_png,
                )
                sink.write(new_sample)
                count += 1
            except Exception as e:
                print(f"[WARN] Skipping sample {sample.get('__key__','?')} in {in_tar.name}: {e}")
    print(f"[OK] {in_tar.name} -> {out_tar.name} | {count} samples")

def main():
    ap = argparse.ArgumentParser(description="Remap all shards in a directory to 14-class mapping and add one-hot.")
    ap.add_argument("--in_dir", required=True, help="Input directory containing *.tar shards.")
    ap.add_argument("--out_dir", required=True, help="Output directory for remapped shards.")
    ap.add_argument("--recursive", type=lambda s: s.lower() in ("1","true","yes","y"),
                    default=True, help="Recurse into subdirectories (default: true).")
    ap.add_argument("--onehot_dtype", choices=["uint8", "float32"], default="uint8",
                    help="One-hot dtype (default: uint8).")
    ap.add_argument("--compression", type=int, default=3,
                    help="PNG compression 0-9 (default: 3).")
    ap.add_argument("--keep_orig_png", type=lambda s: s.lower() in ("1","true","yes","y"),
                    default=False, help="Also store *_rgb_orig.png.")
    ap.add_argument("--overwrite", type=lambda s: s.lower() in ("1","true","yes","y"),
                    default=False, help="Overwrite existing output shards.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    pattern = "**/*.tar" if args.recursive else "*.tar"
    shards = sorted(in_dir.glob(pattern))
    if not shards:
        raise SystemExit(f"No .tar shards found in {in_dir} (recursive={args.recursive}).")

    for in_tar in shards:
        rel = in_tar.relative_to(in_dir)
        out_tar = out_dir / rel
        process_shard(
            in_tar=in_tar,
            out_tar=out_tar,
            onehot_dtype=args.onehot_dtype,
            compression=args.compression,
            keep_orig_png=args.keep_orig_png,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()