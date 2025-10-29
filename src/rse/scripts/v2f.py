import cv2
import os
from pathlib import Path
import shutil

def main():
    
    # 1. Path to your source video
    INPUT_VIDEO = "/media/slsecret/T7/mcgill/b.mp4"
    
    # --- Automatically determine output directory ---
    video_path = Path(INPUT_VIDEO)
    
    # 2. Path to the folder where frames will be saved
    # This directory will be created in the same folder as the video.
    # The folder name will be the video's name + "_frames".
    # e.g., "a.mp4" -> "a_frames/"
    OUTPUT_FRAME_DIR = video_path.parent / f"{video_path.stem}_frames"
    
    # --- END OF PARAMETERS ---

    print(f"Starting frame extraction for: {video_path.resolve()}")
    output_dir = Path(OUTPUT_FRAME_DIR)

    # Clean up the directory if it already exists
    if output_dir.exists():
        print(f"Cleaning up old directory: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            print(f"Error removing directory {output_dir}: {e}")
            return

    # Create the new, empty directory
    print(f"Creating new directory: {output_dir}")
    output_dir.mkdir(parents=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {INPUT_VIDEO}")
        return

    frame_count = 0
    print("Extracting frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use 6-digit padding for correct file sorting (000001.jpg, 000002.jpg, etc.)
        frame_filename = str(output_dir / f"{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  ... extracted {frame_count} frames")

    cap.release()
    print(f"\nExtraction complete.")
    print(f"Total frames: {frame_count}")
    print(f"Frames saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()