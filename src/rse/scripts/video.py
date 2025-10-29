#!/usr/bin/env python3
import os
import cv2
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

MODEL_TYPE = 'pidnet-s'
USE_CITYSCAPES = True

cur_dir = Path(__file__).parent.resolve()
model_path = cur_dir / "../pretrained_models/cityscapes/best0930.pt"
PRETRAINED_MODEL_PATH = str(model_path)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

COLOR_MAP = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32), (255, 0, 255), (0, 255, 255)
]

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    print(f"Loaded {len(pretrained_dict)} parameters!")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= MEAN
    image /= STD
    return image

def run_segmentation(img, model, device):
    resized = False
    old_shape = img.shape
    
    if img.shape[:2] != (1024, 2048):
        img_resized = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_LINEAR)
        resized = True
    else:
        img_resized = img

    img_input = input_transform(img_resized).transpose((2, 0, 1))
    img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_input)
        pred = F.interpolate(pred, size=img_resized.shape[:2], mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    sv_img = np.zeros_like(img_resized).astype(np.uint8)
    for i, color in enumerate(COLOR_MAP):
        for j in range(3):
            sv_img[:, :, j][pred == i] = color[j]

    if resized:
        sv_img = cv2.resize(sv_img, (old_shape[1], old_shape[0]), interpolation=cv2.INTER_LINEAR)
        
    return sv_img

def process_video(input_path, output_path, model, device):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {input_path}")
    print(f"Properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, {frame_count} frames")

    for _ in tqdm(range(frame_count), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        segmented_frame = run_segmentation(frame, model, device)
        out.write(segmented_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nSuccessfully saved segmented video to: {output_path}")

def main():
    input_path = "/home/slsecret/Downloads/VID_20251027_153913.mp4"
    output_path = "/home/slsecret/Downloads/VID_20251027_153913_segmented.mp4"
    model_path = PRETRAINED_MODEL_PATH

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU (CUDA)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print(f"Loading model: {MODEL_TYPE}")
    model = models.pidnet.get_pred_model(MODEL_TYPE, 21 if USE_CITYSCAPES else 11)
    model = load_pretrained(model, model_path).to(device)
    model.eval()
    print("Model loaded successfully.")

    process_video(input_path, output_path, model, device)

if __name__ == '__main__':
    main()