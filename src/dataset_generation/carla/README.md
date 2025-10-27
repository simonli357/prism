
# GridMap WebDataset (PNG + NPY) — Quick Reference

This dataset stores synchronized pairs of ground-truth (GT) and training (TR) grid maps in **WebDataset** shards, with **PNG** for RGB and **NPY** for elevation. It’s designed to be practical for PyTorch (including temporal models like ConvLSTM) while remaining easy to visualize with OpenCV.

---

## Dataset Layout

- Sharded tar archives:
  - `shard-000000.tar`, `shard-000001.tar`, …
- Each **sample = one synchronized time step** with a unique **key**:
  - `run1_000000123`, `run1_000000124`, …

Inside each shard, every sample contains:

| File          | Type / Shape              | Meaning |
|---------------|---------------------------|---------|
| `gt_rgb.png`  | `uint8` PNG, `[H,W,3]`    | Ground-truth RGB (ready for `cv2.imread`) |
| `tr_rgb.png`  | `uint8` PNG, `[H,W,3]`    | Training/input RGB |
| `gt_elev.npy` | `float32` NumPy, `[H,W]`  | Ground-truth elevation (meters); NaN = unknown |
| `tr_elev.npy` | `float32` NumPy, `[H,W]`  | Training/input elevation |
| `valid.png`   | `uint8` PNG, `[H,W]`      | (Optional) mask where GT elevation is finite (0/255) |
| `meta.json`   | JSON                      | Per-frame metadata (see below) |

**Notes**
- All images are **square** and spatially aligned (center-cropped to the same side length).
- RGB is 8-bit; elevation is `float32` with NaNs indicating unknown cells.

### `meta.json` example

```json
{
  "session": "run1",
  "h": 256,
  "w": 256,
  "t_gt": 1234.567,
  "t_tr": 1234.602,
  "gt_frame": "map",
  "tr_frame": "map",
  "resolution_gt": 0.05,
  "resolution_tr": 0.05
}
```

---

## How to Use

### Quick Visualization (OpenCV)

- **RGB**: `cv2.imread` the PNGs (returns BGR) → convert to RGB if needed.
- **Elevation heatmap**:
  1. Load `*_elev.npy`.
  2. Min–max normalize over finite values.
  3. Colorize (e.g., `cv2.COLORMAP_JET`).
- **Valid mask** (`valid.png`): overlay or use to ignore unknown cells.

### PyTorch Training (with WebDataset)

- Stream shards with `webdataset`:
  - Decode `gt_rgb.png`, `tr_rgb.png` to tensors (`uint8`→`float`/`[0,1]`).
  - Load `gt_elev.npy`, `tr_elev.npy` as `float32` tensors.
  - Use `meta.json` for conditioning (timestamps, frames, resolution).

- **Temporal stacks (ConvLSTM)**:
  - Group by `session` (prefix of the sample key) or maintain an index mapping `session → ordered keys`.
  - Build sequences by taking consecutive samples within the same session:
    - RGB: `[T, H, W, 3]` (or channels-first).
    - Elevation: `[T, H, W]`.
    - Mask (optional): `[T, H, W]`.

### Conventions & Tips

- **Channel order**: stored images are **RGB**; OpenCV loads **BGR** → use `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` when needed.
- **Units**: elevation in meters; NaN = unknown.
- **Resolution**: `resolution_gt` and `resolution_tr` are in meters/cell.
- **Sharding**: keep shards ~1–2 GB for fast training & easy portability.
- **Extensibility**: add future files per sample (e.g., `sem.png`, `normals.npy`) without changing the existing schema.

---

## Why This Format

- **Practical**: RGB is immediately viewable (PNG), elevation is exact (NPY).
- **Efficient**: Sharded tars stream well in PyTorch (multi-worker / distributed).
- **Flexible**: Easy to append per-frame modalities or metadata later.
