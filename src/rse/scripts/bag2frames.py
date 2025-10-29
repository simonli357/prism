#!/usr/bin/env python3

"""
Extracts images from a ROS bag file and saves them as PNGs.
Handles both sensor_msgs/Image and sensor_msgs/CompressedImage.
"""

import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import argparse
import sys

def extract_images(bag_file, topic_name, output_dir, subfolder):
    """
    Extracts images from a single topic in a single bag file.
    """
    
    # --- 1. Setup Environment ---
    print(f"  Processing topic: '{topic_name}'")
    
    # Create the full output path (e.g., .../frames/zed2i_left)
    full_output_dir = os.path.join(output_dir, subfolder)
    os.makedirs(full_output_dir, exist_ok=True)
    
    bridge = CvBridge()
    count = 0
    
    # --- 2. Open Bag and Read Messages ---
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            # We filter by the specific topic
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                
                try:
                    # --- 3. Convert ROS Image to OpenCV Image ---
                    if msg._type == 'sensor_msgs/CompressedImage':
                        # Handle compressed images (e.g., JPEG, PNG)
                        cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                        
                    elif msg._type == 'sensor_msgs/Image':
                        # Handle raw images (e.g., bgr8, 16UC1)
                        # We use 'passthrough' to keep the original encoding (like 16-bit depth)
                        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
                        
                        # If it's a 16-bit depth image, it's fine.
                        # If it's something else we can't save (like 'bayer_rggb8'), cv2.imwrite will fail.
                        
                    else:
                        print(f"  Skipping message of unknown type: {msg._type}")
                        continue

                    # --- 4. Save Image to File ---
                    filename = f"frame_{count:06d}.png"
                    save_path = os.path.join(full_output_dir, filename)
                    
                    # cv2.imwrite can save 8-bit (BGR) and 16-bit (grayscale) images as PNGs
                    cv2.imwrite(save_path, cv_image)
                    
                    count += 1
                    
                    if count % 100 == 0:
                        # Print progress update
                        sys.stdout.write(f"\r  Saved {count} frames...")
                        sys.stdout.flush()

                except CvBridgeError as e:
                    print(f"  CvBridgeError converting image: {e}")
                except Exception as e:
                    print(f"  Error processing message: {e}")

    except rosbag.BagException as e:
        print(f"  Error reading bag file '{bag_file}': {e}")
        return
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")
        return

    if count > 0:
        print(f"\n  Finished. Extracted {count} frames for this topic.")
    else:
        # This is not an error; the topic just wasn't in this bag.
        print("  Topic not found in this bag.")


def main():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument('--bagfile', required=True, help="Path to the .bag file")
    parser.add_argument('--topic', required=True, help="ROS topic name to extract (e.g., /boxi/zed2i/left/image_raw/compressed)")
    parser.add_argument('--output_dir', required=True, help="Base output directory (e.g., .../frames)")
    parser.add_argument('--subfolder', required=True, help="Subfolder name for this topic (e.g., zed2i_left)")
    
    args = parser.parse_args()
    
    extract_images(args.bagfile, args.topic, args.output_dir, args.subfolder)

if __name__ == '__main__':
    main()
