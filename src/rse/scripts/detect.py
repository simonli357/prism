import glob
import os
import cv2
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image

# Configuration Section (change these variables as needed)
MODEL_TYPE = 'pidnet-s'  # Options: 'pidnet-s', 'pidnet-m', 'pidnet-l'
USE_CITYSCAPES = True    # True for Cityscapes pretrained model
PRETRAINED_MODEL_PATH = './pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt'
INPUT_IMAGE_DIR = './samples/'
IMAGE_FORMAT = '.png'
OUTPUT_DIR = os.path.join(INPUT_IMAGE_DIR, 'outputs/')

# Normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

COLOR_MAP = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32)
]

def input_transform(image):
    """Normalize and transform the image"""
    image = image.astype(np.float32)[:, :, ::-1]  # BGR to RGB
    image = image / 255.0
    image -= MEAN
    image /= STD
    return image

def load_pretrained(model, pretrained):
    """Load the pretrained model weights"""
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    print(f"Loaded {len(pretrained_dict)} parameters!")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

def process_images():
    """Main function to process images"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Model
    model = models.pidnet.get_pred_model(MODEL_TYPE, 19 if USE_CITYSCAPES else 11)
    model = load_pretrained(model, PRETRAINED_MODEL_PATH).cuda()
    model.eval()

    # Find all images
    images_list = glob.glob(INPUT_IMAGE_DIR + '*' + IMAGE_FORMAT)
    print(f"Processing {len(images_list)} images...")

    with torch.no_grad():
        for img_path in images_list:
            img_name = os.path.basename(img_path)
            print(f"Processing image: {img_name}")

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Error: Image not found - {img_path}")
                continue

            resized = False
            old_shape = img.shape
            if img.shape[:2] != (1024, 2048):
                img = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_LINEAR)
                resized = True

            # Transform and predict
            img_input = input_transform(img).transpose((2, 0, 1))
            img_input = torch.from_numpy(img_input).unsqueeze(0).cuda()
            pred = model(img_input)
            pred = F.interpolate(pred, size=img.shape[:2], mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            # Save the output image
            sv_img = np.zeros_like(img).astype(np.uint8)
            for i, color in enumerate(COLOR_MAP):
                for j in range(3):
                    sv_img[:, :, j][pred == i] = color[j]

            sv_img = Image.fromarray(sv_img)
            if resized:
                sv_img = sv_img.resize((old_shape[1], old_shape[0]), Image.BILINEAR)

            output_path = os.path.join(OUTPUT_DIR, img_name)
            sv_img.save(output_path)
            print(f"Saved output to: {output_path}")

if __name__ == '__main__':
    process_images()
