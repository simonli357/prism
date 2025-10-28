#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import io
import json
import argparse
import numpy as np
import cv2
import rospy
import message_filters
from typing import Optional, Tuple
from grid_map_msgs.msg import GridMap
from rospy.numpy_msg import numpy_msg
import std_msgs.msg as std_msgs
from message_filters import SimpleFilter

import webdataset as wds  # pip install webdataset


# -------------------------- Utilities --------------------------
class _HeaderProxy:
    """Expose msg.info.header as msg.header for message_filters, delegate everything else."""
    __slots__ = ("_msg", "header")
    def __init__(self, msg):
        self._msg = msg
        self.header = getattr(getattr(msg, "info", None), "header", std_msgs.Header())
    def __getattr__(self, name):
        return getattr(self._msg, name)

class HeaderAdapter(SimpleFilter):
    """Wrap a Subscriber and re-emit messages as _HeaderProxy objects."""
    def __init__(self, sub):
        super(HeaderAdapter, self).__init__()
        sub.registerCallback(self._cb)
    def _cb(self, msg):
        self.signalMessage(_HeaderProxy(msg))
def reshape_layer_to_2d(f32_flat, layout, outer_start, inner_start) -> np.ndarray:
    """Reconstruct (rows, cols) and undo GridMap circular buffer offsets."""
    cols = int(layout.dim[0].size)
    rows = int(layout.dim[1].size)
    arr  = np.asarray(f32_flat, dtype=np.float32, order='C')
    arr2d = arr.reshape((cols, rows), order='C').T  # -> (rows, cols)
    if inner_start:
        arr2d = np.roll(arr2d, -int(inner_start), axis=0)
    if outer_start:
        arr2d = np.roll(arr2d, -int(outer_start), axis=1)
    return arr2d

def get_layer_2d(msg: GridMap, name: str) -> Optional[np.ndarray]:
    try:
        idx = msg.layers.index(name)
    except ValueError:
        rospy.logwarn("Layer '%s' not found. Available: %s", name, list(msg.layers))
        return None
    layer = msg.data[idx]
    return reshape_layer_to_2d(
        layer.data, layer.layout,
        outer_start=msg.outer_start_index,
        inner_start=msg.inner_start_index,
    )

def unpack_rgb_from_float_np(f_rgb: np.ndarray):
    """
    Robustly unpack bit-packed RGB from a float32 array by reading raw bytes.
    Little-endian: byte0=b, byte1=g, byte2=r, byte3=exp/sign.
    """
    f = np.asarray(f_rgb, dtype=np.float32, order='C')
    H, W = f.shape
    bytes4 = f.view(np.uint8).reshape(H, W, 4)
    b = bytes4[..., 0].copy()
    g = bytes4[..., 1].copy()
    r = bytes4[..., 2].copy()
    return r, g, b

def center_crop_square(arr: np.ndarray) -> np.ndarray:
    """Center-crop a [H,W] or [H,W,C] array to a square."""
    if arr.ndim == 2:
        H, W = arr.shape
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        return arr[y0:y0+side, x0:x0+side]
    elif arr.ndim == 3:
        H, W = arr.shape[:2]
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        return arr[y0:y0+side, x0:x0+side, :]
    return arr

def build_hw4(rgb_layer_2d: np.ndarray, elev_layer_2d: np.ndarray) -> Optional[np.ndarray]:
    """Build [H, W, 4] array with channels [R, G, B, Elevation], then square-crop."""
    if rgb_layer_2d is None or elev_layer_2d is None:
        return None
    H = min(rgb_layer_2d.shape[0], elev_layer_2d.shape[0])
    W = min(rgb_layer_2d.shape[1], elev_layer_2d.shape[1])
    def crop_to(Ht, Wt, a):
        y0 = (a.shape[0] - Ht) // 2
        x0 = (a.shape[1] - Wt) // 2
        return a[y0:y0+Ht, x0:x0+Wt]
    rgb_layer_2d = crop_to(H, W, rgb_layer_2d)
    elev_layer_2d = crop_to(H, W, elev_layer_2d)

    r, g, b = unpack_rgb_from_float_np(rgb_layer_2d)
    hw4 = np.zeros((H, W, 4), dtype=np.float32)
    hw4[..., 0] = r.astype(np.float32)
    hw4[..., 1] = g.astype(np.float32)
    hw4[..., 2] = b.astype(np.float32)
    hw4[..., 3] = elev_layer_2d.astype(np.float32, copy=False)
    return center_crop_square(hw4)

def to_png_bytes_rgb(rgb_uint8_hwc: np.ndarray) -> bytes:
    """Encode RGB [H,W,3] uint8 -> PNG bytes (OpenCV expects BGR)."""
    assert rgb_uint8_hwc.dtype == np.uint8
    bgr = cv2.cvtColor(rgb_uint8_hwc, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()

def elev_valid_mask(elev_f32: np.ndarray) -> np.ndarray:
    """0/255 uint8 mask where elevation is finite."""
    m = np.isfinite(elev_f32)
    return (m.astype(np.uint8) * 255)

def make_mosaic(gt_rgb: np.ndarray, tr_rgb: np.ndarray, gt_elev: np.ndarray, tr_elev: np.ndarray) -> np.ndarray:
    """For --show: 2x2 visualization (RGB top, elevation heatmaps bottom)."""
    def heat(e):
        mask = np.isfinite(e)
        if not np.any(mask): return np.zeros((*e.shape,3), np.uint8)
        e2 = np.zeros_like(e, np.float32); e2[mask]=(e[mask]-np.nanmin(e[mask]))/(np.nanmax(e[mask])-np.nanmin(e[mask])+1e-6)
        return cv2.applyColorMap((e2*255).astype(np.uint8), cv2.COLORMAP_JET)
    gtrgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
    trrgb = cv2.cvtColor(tr_rgb, cv2.COLOR_RGB2BGR)
    ge = heat(gt_elev); te = heat(tr_elev)
    return np.vstack([np.hstack([gtrgb,trrgb]), np.hstack([ge,te])])


# -------------------------- Node --------------------------

class GridMapToWebDataset:
    def __init__(self, args):
        # Params
        self.slop_sec = float(rospy.get_param("~approx_sync_slop", 0.05))
        self.queue_size = int(rospy.get_param("~queue_size", 20))
        self.include_mask = bool(rospy.get_param("~include_mask", True))
        self.session = str(rospy.get_param("~session", "run1"))
        self.show = args.show if args.show is not None else bool(rospy.get_param("~show", False))

        # WebDataset writer params
        self.wds_pattern = str(rospy.get_param("~wds_pattern", "/media/slsecret/T7/carla3/data/town7b/gridmap_wds/shard-%06d.tar"))
        self.wds_maxcount = int(rospy.get_param("~wds_maxcount", 300))
        self.wds_maxsize_mb = int(rospy.get_param("~wds_maxsize_mb", 1024))  # 0 = ignore
        os.makedirs(os.path.dirname(self.wds_pattern), exist_ok=True)

        # Writer (only if not showing)
        self.writer = None
        if not self.show:
            kw = dict(maxcount=self.wds_maxcount)
            if self.wds_maxsize_mb and self.wds_maxsize_mb > 0:
                kw["maxsize"] = self.wds_maxsize_mb * 1024 * 1024
            self.writer = wds.ShardWriter(self.wds_pattern, **kw)

        gt_sub = message_filters.Subscriber(
            "/elevation_mapping/elevation_map_raw", numpy_msg(GridMap), queue_size=self.queue_size
        )
        tr_sub = message_filters.Subscriber(
            "/elevation_mapping1/elevation_map_raw", numpy_msg(GridMap), queue_size=self.queue_size
        )

        # Adapt so each message has a visible .header (copied from info.header)
        gt_adapt = HeaderAdapter(gt_sub)
        tr_adapt = HeaderAdapter(tr_sub)

        # Timestamp-based sync (no arrival-time fallback)
        ats = message_filters.ApproximateTimeSynchronizer(
            [gt_adapt, tr_adapt],
            queue_size=self.queue_size,
            slop=self.slop_sec,
            allow_headerless=False,
        )
        ats.registerCallback(self.callback)

        self.counter = 0
        rospy.loginfo("GridMapToWebDataset initialized. show=%s, pattern=%s, maxcount=%d, slop=%.3f",
                    self.show, self.wds_pattern, self.wds_maxcount, self.slop_sec)

    def _crop_pair_to_common_square(self, a: np.ndarray, b: np.ndarray):
        side = min(a.shape[0], a.shape[1], b.shape[0], b.shape[1])
        def crop(x):
            H, W = x.shape[:2]
            y0 = (H - side) // 2
            x0 = (W - side) // 2
            return x[y0:y0+side, x0:x0+side, :]
        return crop(a), crop(b)

    def callback(self, gt_msg: GridMap, tr_msg: GridMap):
        # Extract layers → build [H,W,4]
        # gt_hw4 = build_hw4(get_layer_2d(gt_msg, "rgb"), get_layer_2d(gt_msg, "elevation"))
        gt_hw4 = build_hw4(get_layer_2d(gt_msg, "rgb"), get_layer_2d(gt_msg, "inpaint"))
        tr_hw4 = build_hw4(get_layer_2d(tr_msg, "rgb"), get_layer_2d(tr_msg, "elevation"))
        tr_inpaint_raw = get_layer_2d(tr_msg, "inpaint")
        
        if gt_hw4 is None or tr_hw4 is None or tr_inpaint_raw is None:
            rospy.logwarn("Missing required layers; skipping frame.")
            return

        # Align both to common square size
        gt_hw4, tr_hw4 = self._crop_pair_to_common_square(gt_hw4, tr_hw4)

        # ADDED: Crop the raw inpaint layer to match the final size of the others
        side = gt_hw4.shape[0]  # They are now all square with the same side length
        h_inpaint, w_inpaint = tr_inpaint_raw.shape
        y0 = (h_inpaint - side) // 2
        x0 = (w_inpaint - side) // 2
        tr_inpaint = tr_inpaint_raw[y0:y0+side, x0:x0+side].astype(np.float32)

        # Split channels
        gt_rgb = gt_hw4[..., :3].astype(np.uint8)
        tr_rgb = tr_hw4[..., :3].astype(np.uint8)
        gt_elev = gt_hw4[..., 3].astype(np.float32)
        tr_elev = tr_hw4[..., 3].astype(np.float32)

        if self.show:
            mosaic = make_mosaic(gt_rgb, tr_rgb, gt_elev, tr_elev)
            # (Optional) overlay stamps if present
            try:
                t_gt = gt_msg.info.header.stamp.to_sec(); t_tr = tr_msg.info.header.stamp.to_sec()
                txt = f"GT: {t_gt:.3f}s | TR: {t_tr:.3f}s"
            except Exception:
                txt = "Headerless sync (arrival time)"
            cv2.putText(mosaic, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("GridMap (RGB / Elevation)", mosaic)
            if cv2.waitKey(1) in (27, ord('q')):
                rospy.signal_shutdown("User requested quit.")
            return

        # --------- Write one WebDataset sample ----------
        key = f"{self.session}_{self.counter:09d}"
        self.counter += 1

        # Encode PNGs
        gt_rgb_png = to_png_bytes_rgb(gt_rgb)
        tr_rgb_png = to_png_bytes_rgb(tr_rgb)

        # Elevation .npy bytes
        # (Use np.save to an in-memory buffer to ensure standard .npy format)
        gt_buf = io.BytesIO(); np.save(gt_buf, gt_elev, allow_pickle=False); gt_npy = gt_buf.getvalue()
        tr_buf = io.BytesIO(); np.save(tr_buf, tr_elev, allow_pickle=False); tr_npy = tr_buf.getvalue()
        tr_inpaint_buf = io.BytesIO(); np.save(tr_inpaint_buf, tr_inpaint, allow_pickle=False); tr_inpaint_npy = tr_inpaint_buf.getvalue()

        # Optional valid mask from elevation
        maybe_mask = None
        if self.include_mask:
            mask = elev_valid_mask(gt_elev)  # using GT; could also save both
            ok, m_buf = cv2.imencode(".png", mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if ok: maybe_mask = m_buf.tobytes()

        # Metadata (keep concise but sufficient to reconstruct)
        t_gt = gt_msg.header.stamp.to_sec()
        t_tr = tr_msg.header.stamp.to_sec()
        meta = dict(
            session=self.session,
            h=int(gt_rgb.shape[0]), w=int(gt_rgb.shape[1]),
            t_gt=t_gt, t_tr=t_tr,
            gt_frame=getattr(gt_msg.header, "frame_id", ""),
            tr_frame=getattr(tr_msg.header, "frame_id", ""),
            resolution_gt=float(getattr(getattr(gt_msg, "info", None), "resolution", float("nan"))),
            resolution_tr=float(getattr(getattr(tr_msg, "info", None), "resolution", float("nan"))),
        )
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

        sample = {
            "__key__": key,
            "gt_rgb.png": gt_rgb_png,
            "tr_rgb.png": tr_rgb_png,
            "gt_elev.npy": gt_npy,
            "tr_elev.npy": tr_npy,
            "meta.json": meta_bytes,
            "tr_inpaint.npy": tr_inpaint_npy,
        }
        # if maybe_mask is not None:
        #     sample["valid.png"] = maybe_mask

        self.writer.write(sample)
        rospy.loginfo("WDS write: %s (H=W=%d)", key, gt_rgb.shape[0])

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

# -------------------------- Main --------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Write synchronized GridMap pairs into WebDataset shards.")
    parser.add_argument("--show", action="store_true", help="Visualize instead of writing shards.")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable visualization.")
    parser.set_defaults(show=None)  # None => defer to ROS param ~show
    args, _ = parser.parse_known_args(rospy.myargv(argv=None)[1:])
    return args

def main():
    args = parse_args()
    rospy.init_node("gridmap_to_webdataset", anonymous=True)
    node = GridMapToWebDataset(args)
    try:
        rospy.spin()
    finally:
        node.close()
        if node.show:
            try: cv2.destroyAllWindows()
            except Exception: pass

if __name__ == "__main__":
    main()