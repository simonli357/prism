#!/usr/bin/env python3
import os
import cv2
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
import rospy
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CompressedImage 
from cv_bridge import CvBridge
import time
import yaml
from pathlib import Path
import math 

SHOW_RGB = True
MODEL_TYPE = 'pidnet-s'  # Options: 'pidnet-s', 'pidnet-m', 'pidnet-l'
USE_CITYSCAPES = True    # True for Cityscapes pretrained model

# Get the directory of the current script
cur_dir = Path(__file__).parent.resolve()
# model_path = cur_dir / f"../pretrained_models/cityscapes/PIDNet_{MODEL_TYPE[-1].upper()}_Cityscapes_test.pt"
model_path = cur_dir / "../pretrained_models/cityscapes/best0930.pt"
PRETRAINED_MODEL_PATH = str(model_path)
SAVE_DIR = str(cur_dir / '../saved_images/')

# Normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# cityscapes color map and classes
COLOR_MAP = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32), (255, 0, 255), (0, 255, 255)
]
CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'snow', 'water'
]


class SegmentationNode:
    def __init__(self):
        rospy.init_node('segmentation_node')
        self.bridge = CvBridge()

        # Load topics from a YAML file passed as a ROS param
        # config_file_name = rospy.get_param('~config', 'topics.yaml')
        config_file_name = rospy.get_param('~config', 'topics_anymal.yaml')
        self.topic_inputs, self.topic_outputs = self.load_topics_from_config(config_file_name)

        if not self.topic_inputs:
             rospy.logerr("No input topics loaded. Shutting down.")
             return

        if torch.cuda.is_available():
            print("GPU is available. Using GPU.")
            self.device = torch.device('cuda')
        else:
            print("GPU is not available. Using CPU.")
            self.device = torch.device('cpu')

        # Load model
        self.model = models.pidnet.get_pred_model(MODEL_TYPE, 21 if USE_CITYSCAPES else 11)
        self.model = self.load_pretrained(self.model, PRETRAINED_MODEL_PATH).to(self.device)
        self.model.eval()

        # DYNAMICALLY create subscribers, publishers, and image storage
        self.subscribers = {}
        self.publishers = {}
        # Dictionaries are now initialized dynamically based on the config file
        self.images = {key: None for key in self.topic_inputs.keys()}
        self.bgr_images = {key: None for key in self.topic_inputs.keys()}

        for key, topic in self.topic_inputs.items():
            # Check if the topic is compressed
            if topic.endswith('/compressed'):
                msg_type = CompressedImage
            else:
                msg_type = ROSImage
            
            self.subscribers[key] = rospy.Subscriber(topic, msg_type, self.callback, callback_args=key, queue_size=1)
            self.publishers[key] = rospy.Publisher(self.topic_outputs[key], ROSImage, queue_size=1)

        self.frame_times = []

    def load_topics_from_config(self, config_file):
        """Loads ROS topics from the YAML config file."""
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / 'config' / config_file
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                inputs = config['ros_topics']['inputs']
                outputs = config['ros_topics']['outputs']
                rospy.loginfo(f"Successfully loaded topics from {config_path}")
                return inputs, outputs
        except FileNotFoundError:
            rospy.logerr(f"Configuration file not found at: {config_path}")
            return {}, {}
        except (yaml.YAMLError, KeyError) as e:
            rospy.logerr(f"Error parsing YAML file or missing key: {e}")
            return {}, {}

    def load_pretrained(self, model, pretrained):
        pretrained_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        print(f"Loaded {len(pretrained_dict)} parameters!")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        return model

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= MEAN
        image /= STD
        return image

    def callback(self, msg, camera):
        """Handles both compressed and raw image messages."""
        try:
            # Check message type and decode accordingly
            if isinstance(msg, CompressedImage):
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error for camera '{camera}': {e}")
            return
            
        self.bgr_images[camera] = cv_image.copy()
        segmented_image = self.segment_image(cv_image)
        self.images[camera] = segmented_image
        self.publish_segmented_image(segmented_image, camera)

    def segment_image(self, img):
        resized = False
        old_shape = img.shape
        if img.shape[:2] != (1024, 2048):
            img = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_LINEAR)
            resized = True

        img_input = self.input_transform(img).transpose((2, 0, 1))
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img_input)
            pred = F.interpolate(pred, size=img.shape[:2], mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        sv_img = np.zeros_like(img).astype(np.uint8)
        for i, color in enumerate(COLOR_MAP):
            for j in range(3):
                sv_img[:, :, j][pred == i] = color[j]

        if resized:
            sv_img = cv2.resize(sv_img, (old_shape[1], old_shape[0]), interpolation=cv2.INTER_LINEAR)
        return sv_img

    def publish_segmented_image(self, img, camera):
        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.publishers[camera].publish(msg)

    def save_images(self, final_display=None):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_folder = os.path.join(SAVE_DIR, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        for cam, img in self.images.items():
            if img is not None:
                save_path = os.path.join(save_folder, f"{cam}.png")
                cv2.imwrite(save_path, img)
                print(f"Saved {cam} image to {save_path}")
        if final_display is not None:
            save_path = os.path.join(save_folder, "combined_display.png")
            cv2.imwrite(save_path, final_display)
            print(f"Saved combined display image to {save_path}")
                
    def _create_image_grid(self, images_dict, grid_size):
        """Creates a grid of images."""
        cam_keys = list(images_dict.keys())
        num_cams = len(cam_keys)
        
        if num_cams == 0:
            return None

        # Determine grid dimensions
        cols = int(math.ceil(math.sqrt(num_cams)))
        rows = int(math.ceil(num_cams / cols))
        
        # Create a blank image to use as a placeholder
        blank_image = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)

        grid_rows = []
        for r in range(rows):
            row_images = []
            for c in range(cols):
                idx = r * cols + c
                if idx < num_cams:
                    key = cam_keys[idx]
                    img = cv2.resize(images_dict[key], grid_size)
                    row_images.append(img)
                else:
                    row_images.append(blank_image)
            grid_rows.append(np.hstack(row_images))
        
        return np.vstack(grid_rows)

    def display_images(self):
        """Displays all available images in an automatically generated grid."""
        grid_size = (512, 256) # W, H for each cell in the grid
        while not rospy.is_shutdown():
            if all(img is not None for img in self.images.values()):
                
                segmented_grid = self._create_image_grid(self.images, grid_size)
                
                if SHOW_RGB and all(img is not None for img in self.bgr_images.values()):
                    bgr_grid = self._create_image_grid(self.bgr_images, grid_size)
                    final_display = np.vstack((bgr_grid, segmented_grid))
                else:
                    final_display = segmented_grid

                cv2.imshow("Segmentation Display", final_display)
                key = cv2.waitKey(1)
                if key == ord('s'):
                    self.save_images(final_display)
                    print("Images saved!")
            
            rospy.sleep(0.01)

if __name__ == '__main__':
    try:
        node = SegmentationNode()
        if not rospy.is_shutdown():
             node.display_images()
             rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down segmentation node.")