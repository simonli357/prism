#!/usr/bin/env python3

import rosbag
import cv2
import os
import argparse
import sys
from cv_bridge import CvBridge, CvBridgeError
import atexit

# Use the .bag submodule for the BagException
import rosbag.bag 

# Global progress tracking
total_frames_processed = 0
last_reported_milestone = 0

def print_progress():
    """Prints the final count of processed frames upon script exit."""
    global total_frames_processed
    if total_frames_processed > 0:
        # Clear the progress line (e.g., "Saved 1200 frames...")
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
        print(f"  Finished. Extracted {total_frames_processed} frames for this topic.")
    else:
        # This will be printed if the topic was found but no frames were saved (e.g., --every_n_frames is huge)
        pass 


# Register the cleanup function to be called at script exit
atexit.register(print_progress)

def extract_images(bag_file, topic_name, output_base_dir, subfolder, every_n_frames):
    """
    Extracts images from a specific topic in a bag file.
    Handles both sensor_msgs/Image and sensor_msgs/CompressedImage.
    """
    global total_frames_processed
    global last_reported_milestone
    total_frames_processed = 0
    last_reported_milestone = 0
    
    bridge = CvBridge()
    
    output_dir = os.path.join(output_base_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True) # Use exist_ok=True

    print(f"  Processing topic: '{topic_name}'")
    
    try:
        try:
            with rosbag.bag.Bag(bag_file, 'r') as bag:
                topic_found = False
                frame_counter = -1 # Start at -1 so the first frame is 0
                
                for topic, msg, t in bag.read_messages(topics=[topic_name]):
                    topic_found = True
                    frame_counter += 1
                    
                    # --- NEW: Skip-frame logic ---
                    # (frame_counter % every_n_frames) == 0 saves frame 0, 2, 4, etc. for N=2
                    if frame_counter % every_n_frames != 0:
                        continue
                    # --- End New Logic ---
                    
                    try:
                        # Determine message type and get OpenCV image
                        if msg._type == 'sensor_msgs/CompressedImage':
                            cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                        elif msg._type == 'sensor_msgs/Image':
                            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                        else:
                            print(f"  Skipping message with unknown type: {msg._type}")
                            continue

                        if cv_image is None:
                            print(f"  Warning: Failed to decode frame {total_frames_processed}")
                            continue

                        # Format filename with timestamp for uniqueness
                        timestamp_sec = t.secs
                        timestamp_nsec = t.nsecs
                        filename = f"{timestamp_sec}_{timestamp_nsec:09d}.png"
                        filepath = os.path.join(output_dir, filename)
                        
                        cv2.imwrite(filepath, cv_image)
                        
                        total_frames_processed += 1
                        
                        # Print progress every 100 frames
                        if total_frames_processed - last_reported_milestone >= 100:
                            sys.stdout.write(f"\r  Saved {total_frames_processed} frames...")
                            sys.stdout.flush()
                            last_reported_milestone = total_frames_processed

                    except CvBridgeError as e:
                        print(f"\n  CvBridge Error: {e}")
                    except cv2.error as e:
                        print(f"\n  OpenCV Error: {e}")
                    except Exception as e:
                        print(f"\n  Error processing frame: {e}")

                if not topic_found:
                    print("  Topic not found in this bag.")
        
        except rosbag.bag.BagException as e:
            print(f"  Error opening or reading bag file '{os.path.basename(bag_file)}': {e}")
            
    except Exception as e:
        print(f"  An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extracts images from a bag file.")
    parser.add_argument('--bagfile', required=True, help="Path to the bag file.")
    parser.add_argument('--topic', required=True, help="Image topic to extract (e.g., /camera/image_raw/compressed).")
    parser.add_argument('--output_dir', required=True, help="Base directory to save the 'frames' folder.")
    parser.add_argument('--subfolder', required=True, help="Subfolder name within 'frames' (e.g., 'zed_left').")
    # --- NEW: Added new argument ---
    parser.add_argument('--every_n_frames', type=int, default=1, help="Save every Nth frame (e.g., 2 to save every 2nd frame).")
    
    args = parser.parse_args()
    
    if args.every_n_frames < 1:
        print("Error: --every_n_frames must be 1 or greater.")
        sys.exit(1)
        
    extract_images(args.bagfile, args.topic, args.output_dir, args.subfolder, args.every_n_frames)

if __name__ == '__main__':
    main()