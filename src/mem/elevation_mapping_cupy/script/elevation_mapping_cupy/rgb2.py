#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np # Added for numpy operations

BRIDGE = CvBridge()

# +++ NEW CLASS FOR BEV PROCESSING +++
class BEVProcessor:
    """
    Handles the special processing for BEV semantic and depth images
    to erase the ego vehicle's footprint.
    """
    def __init__(self):
        rospy.loginfo("Initializing BEVProcessor...")
        self.EGO_RECT_WIDTH_PX = 62   # Vehicle width in pixels
        self.EGO_RECT_LENGTH_PX = 130  # Vehicle length in pixels
        self.DEPTH_FILL_VALUE = 5.01  # As requested, camera height

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
        """Calculates the bounding box for the centrally-located ego vehicle."""
        center_x, center_y = width // 2, height // 2
        x1 = center_x - self.EGO_RECT_WIDTH_PX // 2
        x2 = center_x + self.EGO_RECT_WIDTH_PX // 2
        y1 = center_y - self.EGO_RECT_LENGTH_PX // 2
        y2 = center_y + self.EGO_RECT_LENGTH_PX // 2
        return int(y1), int(y2), int(x1), int(x2)

    def semantic_callback(self, msg):
        """Erases the ego vehicle from the semantic BEV image by setting pixels to black."""
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
        """Erases the ego vehicle from the depth BEV image by setting pixels to camera height."""
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
        if img.shape[2] == 4:
            return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img3 = img[:, :, :3]
        return cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    except Exception:
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            return img

class RGB8Relay(object):
    def __init__(self, in_topic: str, out_topic: str, q: int = 1):
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.pub = rospy.Publisher(self.out_topic, Image, queue_size=q)
        self.sub = rospy.Subscriber(self.in_topic, Image, self.cb,
                                    queue_size=q, buff_size=2**24)
        rospy.loginfo("RGB8Relay: %s -> %s", self.in_topic, self.out_topic)

    def cb(self, msg: Image):
        enc_lower = (msg.encoding or "").lower()
        img = BRIDGE.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        rgb = to_rgb(img, enc_lower)
        out = BRIDGE.cv2_to_imgmsg(rgb, encoding="rgb8")
        out.header = msg.header
        self.pub.publish(out)

# --- MODIFIED make_pairs FUNCTION ---
def make_pairs():
    """Build (input, output) topics for all cameras EXCEPT the BEV ones."""
    bases = [
        "/carla/ego_vehicle/camera_front",
        "/carla/ego_vehicle/camera_left",
        "/carla/ego_vehicle/camera_rear",
        "/carla/ego_vehicle/camera_right",
        # "/carla/ego_vehicle/camera_bev", # This is a regular RGB camera, can be left in if needed
        "/carla/ego_vehicle/semantic_camera_front",
        "/carla/ego_vehicle/semantic_camera_left",
        "/carla/ego_vehicle/semantic_camera_rear",
        "/carla/ego_vehicle/semantic_camera_right",
        # semantic_camera_bev is now handled by BEVProcessor
        #"/carla/ego_vehicle/semantic_camera_bev", 
        "/carla/ego_vehicle/gt_semantic_camera_ne",
        "/carla/ego_vehicle/gt_semantic_camera_nw",
        "/carla/ego_vehicle/gt_semantic_camera_se",
        "/carla/ego_vehicle/gt_semantic_camera_sw"
    ]
    pairs = []
    for b in bases:
        # Exclude the semantic BEV camera which gets special handling
        if "semantic_camera_bev" not in b:
            pairs.append((f"{b}/image", f"{b}/image_rgb"))
    return pairs

if __name__ == "__main__":
    rospy.init_node("carla_image_processor")

    bev_processor = BEVProcessor()
    
    param_pairs = rospy.get_param("~topics", None)
    if param_pairs:
        pairs = [(str(inp), str(out)) for inp, out in param_pairs]
    else:
        pairs = make_pairs()
    
    relays = [RGB8Relay(inp, out) for inp, out in pairs]
    
    rospy.loginfo("Started %d RGB converters and the BEV Processor.", len(relays))
    rospy.spin()