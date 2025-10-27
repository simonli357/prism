#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Display CARLA ego vehicle cameras in an OpenCV mosaic with intrinsics overlay.

Existing (GT) topics kept:
- /carla/ego_vehicle/gt_camera_ne/camera_info
- /carla/ego_vehicle/gt_camera_ne/image
- /carla/ego_vehicle/gt_camera_nw/camera_info
- /carla/ego_vehicle/gt_camera_nw/image
- /carla/ego_vehicle/gt_camera_se/camera_info
- /carla/ego_vehicle/gt_camera_se/image
- /carla/ego_vehicle/gt_camera_sw/camera_info
- /carla/ego_vehicle/gt_camera_sw/image

ADDED (non-GT) topics:
- /carla/ego_vehicle/camera_front/camera_info
- /carla/ego_vehicle/camera_front/image
- /carla/ego_vehicle/camera_left/camera_info
- /carla/ego_vehicle/camera_left/image
- /carla/ego_vehicle/camera_rear/camera_info
- /carla/ego_vehicle/camera_rear/image
- /carla/ego_vehicle/camera_right/camera_info
- /carla/ego_vehicle/camera_right/image
"""

import sys
import time
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

# ---------- Config ----------
WINDOW_NAME = "CARLA Ego Vehicle Cameras"
TARGET_PANEL_H = 360     # resize each view to this height (keeps aspect)
GRID_COLS = 4            # mosaic columns; 8 cams -> 2 rows x 4 cols by default
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Keep your original GT NE/NW/SE/SW cameras
SEMANTIC = True
semantic_str = "semantic_" if SEMANTIC else ""
CAMERAS = [
    {
        "name": "gt_ne",
        "image_topic": "/carla/ego_vehicle/gt_"+semantic_str+"camera_ne/image",
        "info_topic":  "/carla/ego_vehicle/gt_"+semantic_str+"camera_ne/camera_info",
    },
    {
        "name": "gt_nw",
        "image_topic": "/carla/ego_vehicle/gt_"+semantic_str+"camera_nw/image",
        "info_topic":  "/carla/ego_vehicle/gt_"+semantic_str+"camera_nw/camera_info",
    },
    {
        "name": "gt_se",
        "image_topic": "/carla/ego_vehicle/gt_"+semantic_str+"camera_se/image",
        "info_topic":  "/carla/ego_vehicle/gt_"+semantic_str+"camera_se/camera_info",
    },
    {
        "name": "gt_sw",
        "image_topic": "/carla/ego_vehicle/gt_"+semantic_str+"camera_sw/image",
        "info_topic":  "/carla/ego_vehicle/gt_"+semantic_str+"camera_sw/camera_info",
    },

    # ---------- Newly added non-GT front/left/rear/right ----------
    {
        "name": "front",
        "image_topic": "/carla/ego_vehicle/"+semantic_str+"camera_front/image",
        "info_topic":  "/carla/ego_vehicle/"+semantic_str+"camera_front/camera_info",
    },
    {
        "name": "left",
        "image_topic": "/carla/ego_vehicle/"+semantic_str+"camera_left/image",
        "info_topic":  "/carla/ego_vehicle/"+semantic_str+"camera_left/camera_info",
    },
    {
        "name": "rear",
        "image_topic": "/carla/ego_vehicle/"+semantic_str+"camera_rear/image",
        "info_topic":  "/carla/ego_vehicle/"+semantic_str+"camera_rear/camera_info",
    },
    {
        "name": "right",
        "image_topic": "/carla/ego_vehicle/"+semantic_str+"camera_right/image",
        "info_topic":  "/carla/ego_vehicle/"+semantic_str+"camera_right/camera_info",
    },
    {"name": "bev", "image_topic": "/carla/ego_vehicle/"+semantic_str+"camera_bev/image",
     "info_topic": "/carla/ego_vehicle/"+semantic_str+"camera_bev/camera_info"},
    {"name": "front_rgb", "image_topic": "/carla/ego_vehicle/camera_front/image",
     "info_topic": "/carla/ego_vehicle/camera_front/camera_info"},
]
# ----------------------------


class CameraPanel:
    def __init__(self, name, image_topic, info_topic):
        self.name = name
        self.image_topic = image_topic
        self.info_topic = info_topic
        self.bridge = CvBridge()

        self.last_img = None
        self.last_stamp = None
        self.width = None
        self.height = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.sub_img = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2**24)
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self._info_cb, queue_size=1)

    def _info_cb(self, msg: CameraInfo):
        self.width = msg.width
        self.height = msg.height
        # Intrinsic matrix K = [fx  0 cx; 0 fy cy; 0 0 1]
        if msg.K and len(msg.K) >= 6:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]

    def _image_cb(self, msg: Image):
        try:
            # Let cv_bridge choose best conversion based on encoding
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(1.0, f"[{self.name}] cv_bridge error: {e}")
            return
        self.last_img = cv_img
        self.last_stamp = msg.header.stamp if msg.header else None

    def get_panel(self, target_h: int) -> np.ndarray:
        """Return an annotated/resized panel image; placeholder if no image yet."""
        if self.last_img is not None:
            h, w = self.last_img.shape[:2]
            scale = float(target_h) / max(h, 1)
            new_w = max(int(w * scale), 1)
            resized = cv2.resize(self.last_img, (new_w, target_h), interpolation=cv2.INTER_AREA)
            panel = resized
        else:
            # placeholder based on camera_info size if known
            ph_w = 640 if self.width is None else max(int(self.width * (target_h / float(self.height or self.width))), 320)
            panel = np.zeros((target_h, ph_w, 3), dtype=np.uint8)
            cv2.putText(panel, "Waiting for image…", (20, target_h // 2), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # overlay info bar
        overlay = panel.copy()
        bar_h = 26
        cv2.rectangle(overlay, (0, 0), (panel.shape[1], bar_h), (0, 0, 0), thickness=-1)
        alpha = 0.55
        panel = cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0)

        ts = ""
        if self.last_stamp is not None:
            ts = f"{self.last_stamp.secs}.{str(self.last_stamp.nsecs).zfill(9)[:3]}s"

        info_str = f"{self.name}"
        if self.width and self.height:
            info_str += f"  {self.width}x{self.height}"
        if self.fx and self.fy:
            info_str += f"  fx={self.fx:.1f} fy={self.fy:.1f}"
        if self.cx and self.cy:
            info_str += f"  cx={self.cx:.1f} cy={self.cy:.1f}"
        if ts:
            info_str += f"  t={ts}"

        cv2.putText(panel, info_str, (10, 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return panel


class MosaicViewer:
    def __init__(self, cams_cfg, grid_cols=4):
        self.cams = [CameraPanel(c["name"], c["image_topic"], c["info_topic"]) for c in cams_cfg]
        self.grid_cols = max(1, int(grid_cols))
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(WINDOW_NAME, 1600, 900)

    def _pad_to_width(self, img, width):
        if img.shape[1] >= width:
            return img
        pad = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=img.dtype)
        return np.hstack([img, pad])

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            # Build list of panels
            panels = [cam.get_panel(TARGET_PANEL_H) for cam in self.cams]
            if not panels:
                rate.sleep()
                continue

            # Determine grid size
            n = len(panels)
            cols = self.grid_cols
            rows = (n + cols - 1) // cols

            # Normalize to full grid (pad with blank tiles if needed for clean stacking)
            blank = np.zeros_like(panels[0])
            while len(panels) < rows * cols:
                panels.append(blank.copy())

            # Compute max width per column for tidy alignment
            col_widths = []
            for c in range(cols):
                col_panels = [panels[r * cols + c] for r in range(rows)]
                col_widths.append(max(p.shape[1] for p in col_panels))

            # Assemble rows
            row_imgs = []
            for r in range(rows):
                row_tiles = []
                for c in range(cols):
                    p = panels[r * cols + c]
                    p = self._pad_to_width(p, col_widths[c])
                    row_tiles.append(p)
                row_imgs.append(np.hstack(row_tiles))

            mosaic = np.vstack(row_imgs)

            cv2.imshow(WINDOW_NAME, mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

            rate.sleep()

        cv2.destroyAllWindows()


def main():
    rospy.init_node("carla_ego_multi_cam_viewer", anonymous=True)
    try:
        viewer = MosaicViewer(CAMERAS, grid_cols=GRID_COLS)
        viewer.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
