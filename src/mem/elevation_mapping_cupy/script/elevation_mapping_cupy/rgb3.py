#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import message_filters
try:
    from .plugins.custom.color_mapping import color28_to_color14, debug_print_new_class_counts, debug_print_old_class_counts
except:
    from plugins.custom.color_mapping import color28_to_color14, debug_print_new_class_counts, debug_print_old_class_counts

BRIDGE = CvBridge()

class DepthMaskProcessor:
    """
    Synchronizes a semantic and a depth image. It creates a mask from the
    depth image for pixels closer than a threshold (2.0m) and applies this
    mask to the semantic image. It also masks out a 100x100 square in a
    pre-defined corner for each camera.
    """
    def __init__(self, direction):
        """
        Initializes the processor for a specific camera direction (e.g., 'ne').
        - direction (str): The camera direction suffix ('ne', 'nw', 'se', 'sw').
        """
        self.direction = direction
        self.DEPTH_THRESHOLD = 1.8  # Meters
        self.CORNER_MASK_SIZE = 200 # Pixels for the square corner mask

        rospy.loginfo(f"Initializing DepthMaskProcessor for '{self.direction}' direction...")

        semantic_topic = f"/carla/ego_vehicle/gt_semantic_camera_{self.direction}/image"
        depth_topic = f"/carla/ego_vehicle/gt_camera_depth_{self.direction}/image"
        self.output_topic = f"/carla/ego_vehicle/gt_semantic_camera_{self.direction}/image_masked"

        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.sem_sub = message_filters.Subscriber(semantic_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sem_sub, self.depth_sub], queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.callback)

        rospy.loginfo(f"DepthMaskProcessor for '{self.direction}' is ready.\n"
                      f"  - Subscribing to: {semantic_topic}\n"
                      f"  - Subscribing to: {depth_topic}\n"
                      f"  - Publishing to:  {self.output_topic}")

    def callback(self, sem_msg, depth_msg):
        """
        Callback for synchronized messages. Converts images, applies the depth
        mask, applies the corner mask, and publishes the result.
        """
        try:
            semantic_image = BRIDGE.imgmsg_to_cv2(sem_msg, desired_encoding="bgr8")
            depth_image = BRIDGE.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

            # 1. Apply the depth mask
            mask = depth_image < self.DEPTH_THRESHOLD
            semantic_image[mask] = (0, 0, 0)

            # 2. Apply the corner mask based on camera direction
            h, w, _ = semantic_image.shape
            s = self.CORNER_MASK_SIZE
            if self.direction == 'ne': # Bottom-right
                semantic_image[h-s:h, w-s:w] = (0, 0, 0)
            elif self.direction == 'nw': # Top-right
                semantic_image[0:s, w-s:w] = (0, 0, 0)
            elif self.direction == 'se': # Bottom-left
                semantic_image[h-s:h, 0:s] = (0, 0, 0)
            elif self.direction == 'sw': # Top-left
                semantic_image[0:s, 0:s] = (0, 0, 0)

            out_msg = BRIDGE.cv2_to_imgmsg(semantic_image, encoding="bgr8")
            out_msg.header = sem_msg.header
            self.pub.publish(out_msg)

        except CvBridgeError as e:
            rospy.logerr(f"DepthMaskProcessor CB Error for '{self.direction}': {e}")


class BEVProcessor:
    """
    Handles the special processing for BEV semantic and depth images
    to erase the ego vehicle's footprint.
    """
    def __init__(self):
        rospy.loginfo("Initializing BEVProcessor...")
        self.EGO_RECT_WIDTH_PX = 105
        self.EGO_RECT_LENGTH_PX = 222
        self.DEPTH_FILL_VALUE = 5.01

        self.sem_sub = rospy.Subscriber(
            "/carla/ego_vehicle/semantic_camera_bev/image",
            Image, self.semantic_callback, queue_size=1, buff_size=2**24)
        
        self.depth_sub = rospy.Subscriber(
            "/carla/ego_vehicle/camera_bev_depth/image",
            Image, self.depth_callback, queue_size=1, buff_size=2**24)

        self.sem_pub = rospy.Publisher(
            "/carla/ego_vehicle/semantic_camera_bev/image_rgb",
            Image, queue_size=1)
        self.depth_pub = rospy.Publisher(
            "/carla/ego_vehicle/camera_bev_depth/image_erased",
            Image, queue_size=1)

    def _get_ego_bbox(self, height, width):
        center_x, center_y = width // 2, height // 2
        x1 = center_x - self.EGO_RECT_WIDTH_PX // 2
        x2 = center_x + self.EGO_RECT_WIDTH_PX // 2
        y1 = center_y - self.EGO_RECT_LENGTH_PX // 2
        y2 = center_y + self.EGO_RECT_LENGTH_PX // 2
        return int(y1), int(y2), int(x1), int(x2)

    def semantic_callback(self, msg):
        try:
            cv_image = BRIDGE.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h, w, _ = cv_image.shape
            y1, y2, x1, x2 = self._get_ego_bbox(h, w)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
            out_msg = BRIDGE.cv2_to_imgmsg(cv_image, encoding="bgr8")
            out_msg.header = msg.header
            self.sem_pub.publish(out_msg)
        except CvBridgeError as e:
            rospy.logerr(f"BEVProcessor Semantic CB Error: {e}")

    def depth_callback(self, msg):
        try:
            cv_image = BRIDGE.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            h, w = cv_image.shape
            y1, y2, x1, x2 = self._get_ego_bbox(h, w)
            cv_image[y1:y2, x1:x2] = self.DEPTH_FILL_VALUE
            out_msg = BRIDGE.cv2_to_imgmsg(cv_image, encoding="32FC1")
            out_msg.header = msg.header
            self.depth_pub.publish(out_msg)
        except CvBridgeError as e:
            rospy.logerr(f"BEVProcessor Depth CB Error: {e}")


def to_rgb(img, enc_lower: str):
    """Return an RGB (H,W,3) uint8 image from various encodings."""
    try:
        if len(img.shape) == 2 or img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if 'bgra' in enc_lower and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        if 'rgba' in enc_lower and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if 'bgr' in enc_lower and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if 'rgb' in enc_lower and img.shape[2] == 3:
            return img
        img3 = img[:, :, :3]
        return cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    except Exception:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class RGB8Relay(object):
    def __init__(self, in_topic: str, out_topic: str, q: int = 1, apply_color_mapping: bool = False):
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.apply_mapping = apply_color_mapping # Store the flag
        self.pub = rospy.Publisher(self.out_topic, Image, queue_size=q)
        self.sub = rospy.Subscriber(self.in_topic, Image, self.cb,
                                    queue_size=q, buff_size=2**24)
        rospy.loginfo("RGB8Relay: %s -> %s (Mapping: %s)",
                      self.in_topic, self.out_topic, self.apply_mapping)

    def cb(self, msg: Image):
        try:
            enc_lower = (msg.encoding or "").lower()
            img = BRIDGE.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            rgb = to_rgb(img, enc_lower)
            # debug_print_old_class_counts(rgb)
            if self.apply_mapping:
                rgb = color28_to_color14(rgb)
                # debug_print_new_class_counts(rgb)
            out = BRIDGE.cv2_to_imgmsg(rgb, encoding="rgb8")
            out.header = msg.header
            self.pub.publish(out)
        except (CvBridgeError, Exception) as e:
            # Added a try/except block for robustness, especially for the mapping function
            rospy.logerr(f"RGB8Relay CB Error for '{self.in_topic}': {e}")


def make_pairs():
    """
    Builds (input, output) topic pairs for RGB conversion.
    The gt_semantic_camera topics now point to the masked output from
    the DepthMaskProcessor.
    """
    # Base topics that are converted directly to RGB
    bases = [
        "/carla/ego_vehicle/camera_front",
        "/carla/ego_vehicle/camera_left",
        "/carla/ego_vehicle/camera_rear",
        "/carla/ego_vehicle/camera_right",
        "/carla/ego_vehicle/semantic_camera_front",
        "/carla/ego_vehicle/semantic_camera_left",
        "/carla/ego_vehicle/semantic_camera_rear",
        "/carla/ego_vehicle/semantic_camera_right",
    ]
    pairs = [(f"{b}/image", f"{b}/image_rgb") for b in bases]

    # Topics that are first masked by depth, then converted to RGB
    masked_semantic_bases = [
        "/carla/ego_vehicle/gt_semantic_camera_ne",
        "/carla/ego_vehicle/gt_semantic_camera_nw",
        "/carla/ego_vehicle/gt_semantic_camera_se",
        "/carla/ego_vehicle/gt_semantic_camera_sw"
    ]
    for b in masked_semantic_bases:
        # Input for RGB8Relay is the output of DepthMaskProcessor
        pairs.append((f"{b}/image_masked", f"{b}/image_rgb"))
        
    return pairs

if __name__ == "__main__":
    rospy.init_node("carla_image_processor")

    # Instantiate the BEV processor
    bev_processor = BEVProcessor()
    
    # Instantiate the depth mask processors for each of the four directions
    directions = ['ne', 'nw', 'se', 'sw']
    mask_processors = [DepthMaskProcessor(d) for d in directions]

    # Get topic pairs for RGB conversion
    param_pairs = rospy.get_param("~topics", None)
    if param_pairs:
        pairs = [(str(inp), str(out)) for inp, out in param_pairs]
    else:
        pairs = make_pairs()
    
    semantic_topics_to_map = [
        "/carla/ego_vehicle/semantic_camera_front/image",
        "/carla/ego_vehicle/semantic_camera_left/image",
        "/carla/ego_vehicle/semantic_camera_rear/image",
        "/carla/ego_vehicle/semantic_camera_right/image",
    ]

    # Create the RGB relay converters
    relays = []
    for inp, out in pairs:
        apply_mapping = (inp in semantic_topics_to_map)
        relays.append(RGB8Relay(inp, out, apply_color_mapping=apply_mapping))
    
    
    rospy.loginfo("Started %d RGB converters, %d Depth Mask processors, and the BEV Processor.",
                  len(relays), len(mask_processors))
    rospy.spin()