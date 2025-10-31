#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def trim_frame_directory(input_dir):
    input_path = Path(input_dir).resolve()
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a valid directory.")
        print(f"Path provided: {input_path}")
        return

    output_name = input_path.name + "_trimmed"
    output_path = input_path.parent / output_name

    try:
        print(f"Creating output directory at: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    print("Finding all frames...")
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    all_frame_paths = []
    for ext in extensions:
        all_frame_paths.extend(input_path.glob(ext))
        
    all_frame_paths.sort()
    
    total_frames = len(all_frame_paths)
    if total_frames == 0:
        print(f"Error: No image frames (jpg, png, bmp, etc.) found in {input_path}")
        return
        
    print(f"Found {total_frames} total frames.")

    frames_to_copy = all_frame_paths[::4]
    num_to_copy = len(frames_to_copy)

    print(f"Selecting every {4} frame ({num_to_copy} frames to copy)...")

    copied_count = 0
    try:
        for src_path in tqdm(frames_to_copy, desc="Copying frames"):
            dest_path = output_path / src_path.name
            
            shutil.copy(str(src_path), str(dest_path))
            copied_count += 1

    except Exception as e:
        print(f"\nAn error occurred during copying: {e}")
    finally:
        print(f"\nDone.")
        print(f"Successfully copied {copied_count} of {num_to_copy} selected frames.")
        print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a new directory with half the frames from an input directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "input_dir", 
        type=str,
        nargs='?',
        default="/media/slsecret/T7/grand_tour/frames110308/zed2i_right",
        help="The path to the input directory full of frames.\n"
             "Defaults to '/media/slsecret/T7/mcgill/a_frames' if not provided."
    )
    
    args = parser.parse_args()
    
    trim_frame_directory(args.input_dir)

if __name__ == "__main__":
    main()

