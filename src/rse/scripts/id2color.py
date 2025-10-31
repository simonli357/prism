import numpy as np
from PIL import Image
from pathlib import Path
import sys

PALETTE = {
    0:  (128, 64,128), 1:  (244, 35,232), 2:  ( 70, 70, 70), 3:  (102,102,156),
    4:  (190,153,153), 5:  (153,153,153), 6:  (250,170, 30), 7:  (220,220,  0),
    8:  (107,142, 35), 9:  (152,251,152), 10: ( 70,130,180), 11: (220, 20, 60),
    12: (255,  0,  0), 13: (  0,  0,142), 14: (  0,  0, 70), 15: (  0, 60,100),
    16: (  0, 80,100), 17: (  0,  0,230), 18: (119, 11, 32),
    19: (255,255,255), # snow
    20: (  0, 255,255), # water
    255:(  0,  0,  0)  # ignore
}

INPUT_DIR = Path("/media/slsecret/T7/mapillary/training/unified_ids")
OUTPUT_DIR = INPUT_DIR.parent / f"{INPUT_DIR.name}_color"

# 4. Optional: Add a progress bar with tqdm
# If you don't have tqdm, it will just use a standard iterator
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Install with 'pip install tqdm' for a progress bar.")
    # Create a dummy tqdm function that just returns the iterator
    tqdm = lambda x, **kwargs: x

# --- End of Configuration ---


def convert_id_to_color(id_array, palette):
    """
    Converts a 2D numpy array of label IDs to a 3D numpy array of RGB colors.
    """
    # Create an empty 3-channel (RGB) array for the color image
    # Initialize with the default color (for ID 255 or any missing ID)
    default_color = palette.get(255, (0, 0, 0))
    color_array = np.full((*id_array.shape, 3), default_color, dtype=np.uint8)

    # Use fast NumPy indexing to apply colors
    for label_id, color in palette.items():
        # Find all pixels where the id_array matches the current label_id
        mask = (id_array == label_id)
        
        # Apply the color to those pixels
        color_array[mask] = color
        
    return color_array

def main():
    """
    Main function to run the conversion process.
    """
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found at {INPUT_DIR}")
        sys.exit(1)

    # Create the output directory, (exist_ok=True) prevents errors if it already exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input folder:  {INPUT_DIR}")
    print(f"🎨 Output folder: {OUTPUT_DIR}")

    # Get a list of all .png files in the input directory.
    # Segmentation masks are almost always .png. 
    # Change '*.png' to '*' if you have other file types like .bmp
    image_files = sorted(list(INPUT_DIR.glob('*.png')))

    if not image_files:
        print(f"Warning: No '.png' images found in {INPUT_DIR}.")
        print("If your files have a different extension (e.g., .bmp, .jpg),")
        print("please edit the 'image_files' line in the script.")
        return

    print(f"\nFound {len(image_files)} images to convert. Starting process...")

    # Process each image with a progress bar
    for input_path in tqdm(image_files, desc="Converting"):
        try:
            # 1. Open the label ID image
            id_img = Image.open(input_path)
            
            # 2. Convert to NumPy array. 
            # Segmentation masks are usually single-channel (L-mode or P-mode).
            id_array = np.array(id_img)
            
            # 3. Perform the conversion
            color_array = convert_id_to_color(id_array, PALETTE)
            
            # 4. Convert the color NumPy array back to a PIL Image
            color_img = Image.fromarray(color_array, 'RGB')
            
            # 5. Define the output path
            output_path = OUTPUT_DIR / input_path.name
            
            # 6. Save the new color image
            color_img.save(output_path)
            
        except Exception as e:
            print(f"\nFailed to process {input_path.name}: {e}")

    print(f"\n✨ Done! All images converted and saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()