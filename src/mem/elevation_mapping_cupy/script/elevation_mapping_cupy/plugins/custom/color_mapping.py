# color_mapping.py
import numpy as np

try:
    import torch
except ImportError:
    torch = None
try:
    import cupy as cp
except ImportError:
    cp = None

SEM_CHANNELS = 12

# --- Original Class Definitions and Constants ---
ORIG = [
    (0, "Unlabeled", (0, 0, 0)), (1, "Roads", (128, 64, 128)),
    (2, "SideWalks", (244, 35, 232)), (3, "Building", (70, 70, 70)),
    (4, "Wall", (102, 102, 156)), (5, "Fence", (190, 153, 153)),
    (6, "Pole", (153, 153, 153)), (7, "TrafficLight", (250, 170, 30)),
    (8, "TrafficSign", (220, 220, 0)), (9, "Vegetation", (107, 142, 35)),
    (10, "Terrain", (152, 251, 152)), (11, "Sky", (70, 130, 180)),
    (12, "Pedestrian", (220, 20, 60)), (13, "Rider", (255, 0, 0)),
    (14, "Car", (0, 0, 142)), (15, "Truck", (0, 0, 70)),
    (16, "Bus", (0, 60, 100)), (17, "Train", (0, 80, 100)),
    (18, "Motorcycle", (0, 0, 230)), (19, "Bicycle", (119, 11, 32)),
    (20, "Static", (110, 190, 160)), (21, "Dynamic", (170, 120, 50)),
    (22, "Other", (55, 90, 80)), (23, "Water", (45, 60, 150)),
    (24, "RoadLine", (157, 234, 50)), (25, "Ground", (81, 0, 81)),
    (26, "Bridge", (150, 100, 100)), (27, "RailTrack", (230, 150, 140)),
    (28, "GuardRail", (180, 165, 180)),
]

# ===================================
# New (0..11) merged 12-class schema
# ===================================
NEW_CLASSES = [
    "Unlabeled",   # 0
    "Roads",       # 1  (Roads, RoadLine)
    "SideWalks",   # 2  (SideWalks, Ground)
    "Structure",   # 3  (Building, Wall, Bridge)
    "Barrier",     # 4  (Fence, GuardRail)
    "PoleSign",    # 5  (Pole, TrafficLight, TrafficSign)
    "Vegetation",  # 6
    "Terrain",     # 7
    "Person",      # 8  (Pedestrian, Rider)
    "TwoWheeler",  # 9  (Bicycle, Motorcycle)
    "Vehicle",     # 10 (Car, Truck, Bus, Train)
    "Water",       # 11
]
CLASS_TRAVERSABILITY = np.array([
    0.3,  # 0: Unlabeled
    0.9,  # 1: Roads
    1.0,  # 2: SideWalks
    0.05, # 3: Structure
    0.05, # 4: Barrier
    0.05, # 5: PoleSign
    0.7,  # 6: Vegetation
    0.8,  # 7: Terrain
    0.0,  # 8: Person
    0.1,  # 9: TwoWheeler
    0.1,  # 10: Vehicle
    0.5,  # 11: Water
], dtype=cp.float32)

NEW_CLASSES_VEHICLE_ID = NEW_CLASSES.index("Vehicle")
NEW_CLASSES_UNLABELED_ID = NEW_CLASSES.index("Unlabeled")
NEW_CLASSES_PERSON_ID = NEW_CLASSES.index("Person") 
NEW_CLASSES_TWOWHEELER_ID = NEW_CLASSES.index("TwoWheeler") 

# Palette for new 12 classes (RGB). Tweak as you like.
NEW_PALETTE = np.array([
    (0,   0,   0),     # 0 Unlabeled
    (128, 64,  128),   # 1 Roads
    (244, 35,  232),   # 2 SideWalks
    (70,  70,  70),    # 3 Structure
    (180, 165, 180),   # 4 Barrier
    (220, 220, 0),     # 5 PoleSign
    (107, 142, 35),    # 6 Vegetation
    (152, 251, 152),   # 7 Terrain
    (220, 20,  60),    # 8 Person
    (119, 11,  32),    # 9 TwoWheeler
    (0,   0,   142),   # 10 Vehicle
    (45,  60,  150),   # 11 Water
], dtype=np.uint8)

# ==========================================================
#  1. PRE-COMPUTE NUMPY LOOKUP TABLES (LUTs) FOR EFFICIENCY
# ==========================================================
# LUT for mapping an RGB color triplet to its original class ID (0-28)
RGB_TO_ORIG_ID_LUT = np.zeros((256, 256, 256), dtype=np.int32)
for idx, _, (r, g, b) in ORIG:
    RGB_TO_ORIG_ID_LUT[r, g, b] = idx

# LUT for mapping original class ID (0-28) to new class ID (0-11)
ORIG_TO_NEW_LUT = np.zeros(29, dtype=np.int32)

# --- Unlabeled stays Unlabeled ---
ORIG_TO_NEW_LUT[0] = 0

# --- Roads group (Roads, RoadLine) -> Roads (1)
for k in [1, 24]:
    ORIG_TO_NEW_LUT[k] = 1

# --- SideWalks (2) and Ground (25) -> SideWalks (2)
for k in [2, 25]:
    ORIG_TO_NEW_LUT[k] = 2

# --- Structure: Building, Wall, Bridge -> 3
for k in [3, 4, 26]:
    ORIG_TO_NEW_LUT[k] = 3

# --- Barrier: Fence, GuardRail -> 4
for k in [5, 28]:
    ORIG_TO_NEW_LUT[k] = 4

# --- PoleSign: Pole, TrafficLight, TrafficSign -> 5
for k in [6, 7, 8]:
    ORIG_TO_NEW_LUT[k] = 5

# --- Vegetation -> 6
ORIG_TO_NEW_LUT[9] = 6

# --- Terrain -> 7
ORIG_TO_NEW_LUT[10] = 7

# --- Sky -> Unlabeled (0)
ORIG_TO_NEW_LUT[11] = 0

# --- Person: Pedestrian, Rider -> 8
for k in [12, 13]:
    ORIG_TO_NEW_LUT[k] = 8

# --- TwoWheeler: Bicycle, Motorcycle -> 9
for k in [19, 18]:
    ORIG_TO_NEW_LUT[k] = 9

# --- Vehicle: Car, Truck, Bus, Train -> 10
for k in [14, 15, 16, 17]:
    ORIG_TO_NEW_LUT[k] = 10

# --- Water -> 11
ORIG_TO_NEW_LUT[23] = 11

# --- Misc: Other -> Unlabeled (0)
for k in [22]:
    ORIG_TO_NEW_LUT[k] = 0

# --- Dynamic -> Person (8)
for k in [21]:
    ORIG_TO_NEW_LUT[k] = 8
    
# --- Static, RailTrack -> Structure (3)
for k in [20, 27]:
    ORIG_TO_NEW_LUT[k] = 3

# Cache for storing LUTs converted to PyTorch/CuPy
_backend_luts = {}

def _get_backend_and_device(arr):
    """Return (module, device) where module is one of (np, cp, torch)."""
    if torch is not None and isinstance(arr, torch.Tensor):
        return torch, arr.device
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp, arr.device
    if isinstance(arr, np.ndarray):
        return np, 'cpu'
    raise TypeError(f"Unsupported array type: {type(arr)}")

def _get_lut(lut_name, backend, device):
    """Convert a NumPy LUT to the target backend and cache it."""
    key = (lut_name, str(device))
    if key not in _backend_luts:
        np_lut = globals()[lut_name]
        if backend == np:
            _backend_luts[key] = np_lut
        elif backend == cp:
            with device:
                _backend_luts[key] = cp.asarray(np_lut)
        elif backend == torch:
            _backend_luts[key] = torch.from_numpy(np_lut).to(device)
    return _backend_luts[key]

def _ensure_rgb_uint8(x, backend):
    if backend == torch:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError("rgb_img must be (H,W,3).")
        if x.dtype != torch.uint8:
            x = x.to(torch.uint8)
        return x.contiguous()
    else:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError("rgb_img must be (H,W,3).")
        dtype = cp.uint8 if backend is cp else np.uint8
        if x.dtype != dtype:
            x = x.astype(dtype, copy=False)
        return x

def _rgb_image_to_orig_ids(rgb_img):
    """Convert RGB semantic color image (H,W,3) -> original IDs (H,W)."""
    backend, device = _get_backend_and_device(rgb_img)
    rgb_img = _ensure_rgb_uint8(rgb_img, backend)
    lut = _get_lut('RGB_TO_ORIG_ID_LUT', backend, device)

    if backend == torch:
        r = rgb_img[..., 0].to(torch.long)
        g = rgb_img[..., 1].to(torch.long)
        b = rgb_img[..., 2].to(torch.long)
        return lut[r, g, b]
    else:
        return lut[rgb_img[..., 0], rgb_img[..., 1], rgb_img[..., 2]]

def color28_to_new_indices_and_color(rgb_img):
    """
    Input:  RGB semantic image in ORIGINAL colors (H,W,3), uint8.
            Accepts NumPy / CuPy / Torch arrays.
    Output: (new_ids, new_color) on the same device as input.
    """
    backend, device = _get_backend_and_device(rgb_img)
    orig_ids = _rgb_image_to_orig_ids(rgb_img)

    orig_to_new_lut = _get_lut('ORIG_TO_NEW_LUT', backend, device)
    new_palette_lut = _get_lut('NEW_PALETTE', backend, device)

    new_ids = orig_to_new_lut[orig_ids]
    new_color = new_palette_lut[new_ids]
    return new_ids, new_color

def color28_to_onehot(rgb_img, dtype=np.float32):
    """
    RGB (original palette) -> one-hot (H,W,SEM_CHANNELS) on same device.
    """
    backend, device = _get_backend_and_device(rgb_img)
    new_ids, _ = color28_to_new_indices_and_color(rgb_img)

    num_classes = SEM_CHANNELS
    if backend == torch:
        return torch.nn.functional.one_hot(new_ids.long(), num_classes).to(dtype)
    else:
        eye = backend.eye(num_classes, dtype=dtype)
        return eye[new_ids]

def onehot_to_color(onehot):
    """

    Input:  onehot (H,W,SEM_CHANNELS) or class IDs (H,W)
    Output: RGB color image (H,W,3) uint8 using NEW_PALETTE
    """
    backend, device = _get_backend_and_device(onehot)

    if onehot.ndim == 3 and onehot.shape[-1] == SEM_CHANNELS:
        new_ids = backend.argmax(onehot, axis=-1)
    elif onehot.ndim == 2:
        new_ids = onehot
    else:
        raise ValueError(f"Input must be (H,W,{SEM_CHANNELS}) one-hot or (H,W) class ids.")

    palette = _get_lut('NEW_PALETTE', backend, device)
    if backend == torch:
        return palette[new_ids.long()]
    else:
        return palette[new_ids]

def color28_to_color_new(rgb_img):
    """One-shot: original color image -> new 12-class color image."""
    _, new_col = color28_to_new_indices_and_color(rgb_img)
    return new_col
def onehot_to_traversability(semantic_map):
    """
    Converts a semantic map (one-hot or class indices) to a traversability map.

    Input:
      semantic_map: (..., SEM_CHANNELS) one-hot or (...,) class IDs.
                    Accepts NumPy / CuPy / Torch.
    Output:
      Traversability map (...) float32, on the same device.
    """
    backend, device = _get_backend_and_device(semantic_map)

    trav_lut = _get_lut('CLASS_TRAVERSABILITY', backend, device)

    if semantic_map.ndim >= 1 and semantic_map.shape[-1] == SEM_CHANNELS:
        if backend == torch:
            if semantic_map.dtype != trav_lut.dtype:
                 semantic_map = semantic_map.to(trav_lut.dtype)
            # Use @ for matrix multiplication
            return semantic_map @ trav_lut
        else: # np or cp
            if semantic_map.dtype != trav_lut.dtype:
                 semantic_map = semantic_map.astype(trav_lut.dtype, copy=False)
            # Use .dot for N-D array dot product on last axis
            return semantic_map.dot(trav_lut)
    
    elif semantic_map.ndim >= 2:
        if backend == torch:
            return trav_lut[semantic_map.long()]
        else:
            # Ensure indices are integers for indexing
            return trav_lut[semantic_map.astype(np.int32)]
    
    else:
         raise ValueError(
            f"Input must be (...,{SEM_CHANNELS}) one-hot or (...,) class ids."
         )
         
# -------------------------------
# Backward-compatible wrappers
# -------------------------------
def color28_to_new14_indices_and_color(rgb_img):
    return color28_to_new_indices_and_color(rgb_img)

def color28_to_onehot14(rgb_img, dtype=np.float32):
    return color28_to_onehot(rgb_img, dtype=dtype)

def onehot14_to_color(onehot):
    return onehot_to_color(onehot)

def color28_to_color14(rgb_img):
    return color28_to_color_new(rgb_img)

def onehot14_to_traversability(semantic_map):
    return onehot_to_traversability(semantic_map)