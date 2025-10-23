#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

BRIDGE = CvBridge()

def to_rgb(img, enc_lower: str):
    """Return an RGB (H,W,3) uint8 image from various encodings."""
    try:
        if len(img.shape) == 2 or img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 3 or 4 channels
        if 'bgra' in enc_lower and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        if 'rgba' in enc_lower and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if 'bgr' in enc_lower and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if 'rgb' in enc_lower and img.shape[2] == 3:
            return img  # already RGB

        # Fallbacks
        if img.shape[2] == 4:
            return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Last-ditch: force 3 channels and treat as BGR
        img3 = img[:, :, :3]
        return cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    except Exception:
        # As a final guard, try OpenCV's generic conversion chains
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
        out.header = msg.header  # preserve timing/frame
        self.pub.publish(out)

def make_pairs():
    """Build (input_image_topic, output_image_topic) for all 8 cameras."""
    bases = [
        "/carla/ego_vehicle/camera_front",
        "/carla/ego_vehicle/camera_left",
        "/carla/ego_vehicle/camera_rear",
        "/carla/ego_vehicle/camera_right",
        "/carla/ego_vehicle/camera_bev",
        "/carla/ego_vehicle/semantic_camera_front",
        "/carla/ego_vehicle/semantic_camera_left",
        "/carla/ego_vehicle/semantic_camera_rear",
        "/carla/ego_vehicle/semantic_camera_right",
        "/carla/ego_vehicle/semantic_camera_bev",
        "/carla/ego_vehicle/gt_semantic_camera_ne",
        "/carla/ego_vehicle/gt_semantic_camera_nw",
        "/carla/ego_vehicle/gt_semantic_camera_se",
        "/carla/ego_vehicle/gt_semantic_camera_sw"
    ]
    pairs = []
    for b in bases:
        pairs.append((f"{b}/image", f"{b}/image_rgb"))
    return pairs

if __name__ == "__main__":
    rospy.init_node("strip_alpha_rgb8_multi")

    # Allow overriding list via ROS param if desired
    # e.g., rosparam set /strip_alpha_rgb8_multi/topics "[['/in1','/out1'],['/in2','/out2']]"
    param_pairs = rospy.get_param("~topics", None)
    if param_pairs:
        pairs = [(str(inp), str(out)) for inp, out in param_pairs]
    else:
        pairs = make_pairs()

    relays = [RGB8Relay(inp, out) for inp, out in pairs]

    rospy.loginfo("Started %d RGB converters.", len(relays))
    rospy.spin()
