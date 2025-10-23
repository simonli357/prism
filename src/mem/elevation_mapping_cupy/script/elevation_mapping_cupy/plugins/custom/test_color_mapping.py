#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Import the functions from your previously created color_mapping.py
# Ensure color_mapping.py is in your ROS package (e.g., <pkg>/src) or on PYTHONPATH.
from .color_mapping import (
    color28_to_onehot14,
    onehot14_to_color,
    color28_to_color14,  # optional, for side-by-side
)

class ColorMappingTester:
    def __init__(self):
        self.bridge = CvBridge()
        topic = "/carla/ego_vehicle/semantic_camera_front/image"
        self.sub = rospy.Subscriber(topic, Image, self.cb, queue_size=1)
        rospy.loginfo("Subscribed to %s", topic)

        # OpenCV display setup
        self.win_in = "Input (original)"
        self.win_color14 = "Mapped (14-class color)"
        self.win_roundtrip = "Round-trip (onehot->color)"
        cv2.namedWindow(self.win_in, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_color14, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_roundtrip, cv2.WINDOW_NORMAL)

    def _to_rgb_uint8(self, msg):
        """
        Convert incoming sensor_msgs/Image to uint8 RGB np.array.
        Handles rgb8/rgba8/bgr8/bgra8 and tries a safe fallback.
        """
        enc = msg.encoding.lower() if msg.encoding else ""
        try:
            # Direct conversions via cv_bridge (it will handle bgr<->rgb internally)
            if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
                # Ask bridge to give us RGB8 regardless of source
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                return img
            elif enc in ("mono8",):
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                # Unknown/other encodings: try passthrough then coerce
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img.ndim == 2:  # HxW
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                if img.ndim == 3 and img.shape[2] == 4:
                    # Heuristic: assume BGRA or RGBA; prefer trying BGRA->RGB then RGBA->RGB
                    try:
                        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    except Exception:
                        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                if img.ndim == 3 and img.shape[2] == 3:
                    # Heuristic: many ROS stacks use BGR by default
                    try:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    except Exception:
                        return img.astype(np.uint8)
                # Last resort: ensure uint8 and 3 channels
                img = img.astype(np.uint8)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3 and img.shape[2] not in (3, 4):
                    # squeeze or pad to 3
                    img = img[:, :, :3] if img.shape[2] > 3 else np.repeat(img, 3 // img.shape[2], axis=2)
                return img
        except Exception as e:
            rospy.logwarn("Failed to convert encoding '%s': %s", enc, e)
            # Fallback: attempt raw passthrough then coerce to RGB
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.ndim == 3 and img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            if img.ndim == 3 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    def cb(self, msg: Image):
        # 1) Convert input to RGB uint8
        rgb = self._to_rgb_uint8(msg)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # 2) RGB (original palette) -> one-hot (H,W,14)
        try:
            onehot = color28_to_onehot14(rgb)  # float32 (H,W,14)
        except Exception as e:
            rospy.logerr("color28_to_onehot14 failed: %s", e)
            return

        # 3) One-hot -> color (14-class palette)
        try:
            color_from_onehot = onehot14_to_color(onehot)  # (H,W,3) uint8
        except Exception as e:
            rospy.logerr("onehot14_to_color failed: %s", e)
            return

        # (Optional) Also compute direct color mapping for visual comparison
        try:
            color14_direct = color28_to_color14(rgb)  # (H,W,3) uint8
        except Exception as e:
            rospy.logwarn("color28_to_color14 failed (optional): %s", e)
            color14_direct = color_from_onehot

        # 4) Verify round-trip equality (onehot->color == direct color)
        if color14_direct.shape == color_from_onehot.shape:
            diff = cv2.absdiff(color14_direct, color_from_onehot)
            n_diff = np.count_nonzero(diff)
            if n_diff == 0:
                rospy.loginfo_throttle(2.0, "Round-trip OK: onehot->color matches direct map.")
            else:
                rospy.logwarn_throttle(2.0, "Round-trip mismatch pixels: %d", n_diff)
        else:
            rospy.logwarn_throttle(2.0, "Shape mismatch: direct %s vs onehot->color %s",
                                   color14_direct.shape, color_from_onehot.shape)

        # 5) Show images
        # Note: OpenCV expects BGR for imshow; convert our RGBs for display.
        bgr_in = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr_color14 = cv2.cvtColor(color14_direct, cv2.COLOR_RGB2BGR)
        bgr_roundtrip = cv2.cvtColor(color_from_onehot, cv2.COLOR_RGB2BGR)

        cv2.imshow(self.win_in, bgr_in)
        cv2.imshow(self.win_color14, bgr_color14)
        cv2.imshow(self.win_roundtrip, bgr_roundtrip)
        cv2.waitKey(1)

def main():
    rospy.init_node("test_color_mapping_node", anonymous=True)
    ColorMappingTester()
    rospy.loginfo("test_color_mapping_node is running. Press Ctrl+C to exit.")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
