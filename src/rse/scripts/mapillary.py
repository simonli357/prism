#!/usr/bin/env python3

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from multiprocessing import Pool, cpu_count

BASE_DIR = "/media/slsecret/T7/mapillary/training"
INPUT_DIR = os.path.join(BASE_DIR, "labels")
OUTPUT_DIR_ID = os.path.join(BASE_DIR, "labels_unified_id")
OUTPUT_DIR_COLOR = os.path.join(BASE_DIR, "labels_unified_color")

CS = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4, "pole": 5,
    "traffic_light": 6, "traffic_sign": 7, "vegetation": 8, "terrain": 9, "sky": 10,
    "person": 11, "rider": 12, "car": 13, "truck": 14, "bus": 15, "train": 16,
    "motorcycle": 17, "bicycle": 18, "snow": 19, "water": 20
}
IGNORE = 255

PALETTE = {
    0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70), 3: (102, 102, 156),
    4: (190, 153, 153), 5: (153, 153, 153), 6: (250, 170, 30), 7: (220, 220, 0),
    8: (107, 142, 35), 9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
    12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70), 15: (0, 60, 100),
    16: (0, 80, 100), 17: (0, 0, 230), 18: (119, 11, 32),
    19: (255, 255, 255),
    20: (0, 255, 255),
    255: (0, 0, 0)
}

MAPILLARY_TO_UNIFIED_ID = {
    "Bird": CS["person"],
    "Ground Animal": CS["person"],
    "Curb": CS["sidewalk"],
    "Fence": CS["fence"],
    "Guard Rail": CS["fence"],
    "Barrier": CS["fence"],
    "Wall": CS["wall"],
    "Bike Lane": CS["road"],
    "Crosswalk - Plain": CS["road"],
    "Curb Cut": CS["sidewalk"],
    "Parking": CS["road"],
    "Pedestrian Area": CS["sidewalk"],
    "Rail Track": CS["road"],
    "Road": CS["road"],
    "Service Lane": CS["road"],
    "Sidewalk": CS["sidewalk"],
    "Bridge": CS["building"],
    "Building": CS["building"],
    "Tunnel": CS["building"],
    "Person": CS["person"],
    "Bicyclist": CS["rider"],
    "Motorcyclist": CS["rider"],
    "Other Rider": CS["rider"],
    "Lane Marking - Crosswalk": CS["road"],
    "Lane Marking - General": CS["road"],
    "Mountain": CS["terrain"],
    "Sand": CS["terrain"],
    "Sky": CS["sky"],
    "Snow": CS["snow"],
    "Terrain": CS["terrain"],
    "Vegetation": CS["vegetation"],
    "Water": CS["water"],
    "Banner": CS["traffic_sign"],
    "Bench": CS["pole"],
    "Bike Rack": CS["pole"],
    "Billboard": CS["traffic_sign"],
    "Catch Basin": IGNORE,
    "CCTV Camera": CS["pole"],
    "Fire Hydrant": CS["pole"],
    "Junction Box": CS["pole"],
    "Mailbox": CS["pole"],
    "Manhole": CS["road"],
    "Phone Booth": CS["building"],
    "Pothole": CS["road"],
    "Street Light": CS["pole"],
    "Pole": CS["pole"],
    "Traffic Sign Frame": CS["traffic_sign"],
    "Utility Pole": CS["pole"],
    "Traffic Light": CS["traffic_light"],
    "Traffic Sign (Back)": CS["traffic_sign"],
    "Traffic Sign (Front)": CS["traffic_sign"],
    "Trash Can": CS["pole"],
    "Bicycle": CS["bicycle"],
    "Boat": CS["car"],
    "Bus": CS["bus"],
    "Car": CS["car"],
    "Caravan": CS["truck"],
    "Motorcycle": CS["motorcycle"],
    "On Rails": CS["train"],
    "Other Vehicle": CS["car"],
    "Trailer": CS["truck"],
    "Truck": CS["truck"],
    "Wheeled Slow": CS["car"],
    "Car Mount": IGNORE,
    "Ego Vehicle": IGNORE,
    "Unlabeled": IGNORE,
}

MAPILLARY_V1_2_COLOR_TO_NAME = {
    (165, 42, 42): "Bird",
    (0, 192, 0): "Ground Animal",
    (196, 196, 196): "Curb",
    (190, 153, 153): "Fence",
    (180, 165, 180): "Guard Rail",
    (90, 120, 150): "Barrier",
    (102, 102, 156): "Wall",
    (128, 64, 255): "Bike Lane",
    (140, 140, 200): "Crosswalk - Plain",
    (170, 170, 170): "Curb Cut",
    (250, 170, 160): "Parking",
    (96, 96, 96): "Pedestrian Area",
    (230, 150, 140): "Rail Track",
    (128, 64, 128): "Road",
    (110, 110, 110): "Service Lane",
    (244, 35, 232): "Sidewalk",
    (150, 100, 100): "Bridge",
    (70, 70, 70): "Building",
    (150, 120, 90): "Tunnel",
    (220, 20, 60): "Person",
    (255, 0, 0): "Bicyclist",
    (255, 0, 100): "Motorcyclist",
    (255, 0, 200): "Other Rider",
    (200, 128, 128): "Lane Marking - Crosswalk",
    (255, 255, 255): "Lane Marking - General",
    (64, 170, 64): "Mountain",
    (230, 160, 50): "Sand",
    (70, 130, 180): "Sky",
    (190, 255, 255): "Snow",
    (152, 251, 152): "Terrain",
    (107, 142, 35): "Vegetation",
    (0, 170, 30): "Water",
    (255, 255, 128): "Banner",
    (250, 0, 30): "Bench",
    (100, 140, 180): "Bike Rack",
    (220, 220, 220): "Billboard",
    (220, 128, 128): "Catch Basin",
    (222, 40, 40): "CCTV Camera",
    (100, 170, 30): "Fire Hydrant",
    (40, 40, 40): "Junction Box",
    (33, 33, 33): "Mailbox",
    (100, 128, 160): "Manhole",
    (142, 0, 0): "Phone Booth",
    (70, 100, 150): "Pothole",
    (210, 170, 100): "Street Light",
    (153, 153, 153): "Pole",
    (128, 128, 128): "Traffic Sign Frame",
    (0, 0, 80): "Utility Pole",
    (250, 170, 30): "Traffic Light",
    (192, 192, 192): "Traffic Sign (Back)",
    (220, 220, 0): "Traffic Sign (Front)",
    (140, 140, 20): "Trash Can",
    (119, 11, 32): "Bicycle",
    (150, 0, 255): "Boat",
    (0, 60, 100): "Bus",
    (0, 0, 142): "Car",
    (0, 0, 90): "Caravan",
    (0, 0, 230): "Motorcycle",
    (0, 80, 100): "On Rails",
    (128, 64, 64): "Other Vehicle",
    (0, 0, 110): "Trailer",
    (0, 0, 70): "Truck",
    (0, 0, 192): "Wheeled Slow",
    (32, 32, 32): "Car Mount",
    (120, 10, 10): "Ego Vehicle",
    (0, 0, 0): "Unlabeled",
}

def build_conversion_maps():
    print("Building conversion lookup tables...")
    color_to_id_map = {}
    
    for color, name in MAPILLARY_V1_2_COLOR_TO_NAME.items():
        if name in MAPILLARY_TO_UNIFIED_ID:
            unified_id = MAPILLARY_TO_UNIFIED_ID[name]
        else:
            unified_id = IGNORE
            print(f"Warning: Original class '{name}' not found in MAPILLARY_TO_UNIFIED_ID. Mapping to IGNORE.")
        
        color_to_id_map[color] = unified_id

    id_to_color_map = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        if i in PALETTE:
            id_to_color_map[i] = PALETTE[i]
        else:
            id_to_color_map[i] = PALETTE[IGNORE]

    return color_to_id_map, id_to_color_map

g_color_to_id_map = None
g_id_to_color_map = None

def init_worker(color_map, id_map):
    global g_color_to_id_map, g_id_to_color_map
    g_color_to_id_map = color_map
    g_id_to_color_map = id_map

def process_image_worker(filename):
    try:
        input_path = os.path.join(INPUT_DIR, filename)
        
        bgr_img = cv2.imread(input_path)
        if bgr_img is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            return filename, False
            
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_img.shape
        
        id_mask = np.full((height, width), IGNORE, dtype=np.uint8)

        for (r, g, b), unified_id in g_color_to_id_map.items():
            mask = (rgb_img[:, :, 0] == r) & \
                   (rgb_img[:, :, 1] == g) & \
                   (rgb_img[:, :, 2] == b)
            
            id_mask[mask] = unified_id

        color_mask_rgb = g_id_to_color_map[id_mask]
        color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)

        base_name = os.path.splitext(filename)[0]
        output_name = f"{base_name}.png"
        
        output_path_id = os.path.join(OUTPUT_DIR_ID, output_name)
        output_path_color = os.path.join(OUTPUT_DIR_COLOR, output_name)
        
        cv2.imwrite(output_path_id, id_mask)
        cv2.imwrite(output_path_color, color_mask_bgr) # <-- THIS LINE IS NOW CORRECTED
        
        return filename, True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename, False

def main():
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output (ID):    {OUTPUT_DIR_ID}")
    print(f"Output (Color): {OUTPUT_DIR_COLOR}")
    
    os.makedirs(OUTPUT_DIR_ID, exist_ok=True)
    os.makedirs(OUTPUT_DIR_COLOR, exist_ok=True)
    
    try:
        image_files = [
            f for f in os.listdir(INPUT_DIR)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not image_files:
            print(f"Error: No images found in {INPUT_DIR}.")
            sys.exit(1)
        print(f"Found {len(image_files)} images to process.")
    except FileNotFoundError:
        print(f"Error: Input directory not found: {INPUT_DIR}")
        sys.exit(1)
        
    color_to_id_map, id_to_color_map = build_conversion_maps()

    num_workers = cpu_count()
    print(f"Processing images using {num_workers} workers...")
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(color_to_id_map, id_to_color_map)) as pool:
        results = list(tqdm(pool.imap_unordered(process_image_worker, image_files), total=len(image_files), unit="image"))

    successful_count = sum(1 for _, success in results if success)
    print(f"\nProcessing complete. Successfully processed {successful_count}/{len(image_files)} images.")
    print(f"Saved ID labels saved to: {OUTPUT_DIR_ID}")
    print(f"Saved Color labels saved to: {OUTPUT_DIR_COLOR}")

if __name__ == "__main__":
    main()