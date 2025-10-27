#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import cupy as cp
from collections import deque
import os
from typing import Tuple, Optional
import torch.nn.functional as F
import torch.utils.dlpack as dlpack
import cv2 

from .unet_conv_lstm import UNetConvLSTM
from .deeplabv3plus import DeepLabV3Plus
from .cnn import CNNCorrectionNetSingle
from .color_mapping import (
    color28_to_onehot14, onehot14_to_color, SEM_CHANNELS, 
    onehot14_to_traversability, color14_to_onehot14, 
    debug_print_old_class_counts, debug_print_new_class_counts
)

def _default_weights() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sensible default weights that match the expected shapes:
      conv1: (4, 1, 3, 3)  dilation=1
      conv2: (4, 1, 3, 3)  dilation=2 (same kernel size/shape)
      conv3: (4, 1, 3, 3)  dilation=3 (same kernel size/shape)
      conv_out: (1, 12, 1, 1)
    """
    # Basic 3x3 kernels: blur + Sobel-like edges + Laplacian-ish
    blur = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]], dtype=np.float32)
    blur = blur / blur.sum()

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = sobel_x.T

    lap = np.array([[0,  1, 0],
                    [1, -4, 1],
                    [0,  1, 0]], dtype=np.float32)

    # Stack four 3x3 filters -> (4, 1, 3, 3)
    kset = np.stack([blur, sobel_x, sobel_y, lap], axis=0)[:, None, :, :]  # (4,1,3,3)

    w1 = kset.copy().astype(np.float32)          # (4,1,3,3)
    w2 = kset.copy().astype(np.float32)          # (4,1,3,3)
    w3 = kset.copy().astype(np.float32)          # (4,1,3,3)

    # 12 input channels -> 1 output via 1x1 conv; start as equal-weight average
    w_out = np.ones((1, 12, 1, 1), dtype=np.float32) / 12.0

    return w1, w2, w3, w_out


class TraversabilityFilter(nn.Module):
    def __init__(self, w1, w2, w3, w_out, device="cuda", use_bias=False):
        super().__init__()
        print("TraversabilityFilter device", device)
        self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=0, bias=use_bias)
        self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=0, bias=use_bias)
        self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=0, bias=use_bias)
        self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

        self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
        self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
        self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
        self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

        if use_bias:
            nn.init.zeros_(self.conv1.bias)
            nn.init.zeros_(self.conv2.bias)
            nn.init.zeros_(self.conv3.bias)
            nn.init.zeros_(self.conv_out.bias)

    def forward(self, elevation_cupy: cp.ndarray) -> cp.ndarray:
        elevation_cupy = elevation_cupy.astype(cp.float32)
        
        # Note: We must use the device the filter's weights are on
        elevation = torch.as_tensor(elevation_cupy, device=self.conv1.weight.device)

        with torch.no_grad():
            x = elevation.view(1, 1, elevation.shape[0], elevation.shape[1])
            out1 = self.conv1(x)                       # (1,4,H-2,W-2)
            out2 = self.conv2(x)                       # (1,4,H-4,W-4)
            out3 = self.conv3(x)                       # (1,4,H-6,W-6)

            out1 = out1[:, :, 2:-2, 2:-2]              # -> (1,4,H-6,W-6)
            out2 = out2[:, :, 1:-1, 1:-1]              # -> (1,4,H-6,W-6)

            feats = torch.cat((out1, out2, out3), dim=1)  # (1,12,H-6,W-6)

            y = self.conv_out(feats.abs())             # (1,1,H-6,W-6)
            y = torch.exp(-y)                          # higher = more traversable

            out_cupy = cp.asarray(y)

        return out_cupy  # (1,1,H-6,W-6)

def get_filter_torch(
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    w_out: np.ndarray,
    device: str = "cuda",
    use_bias: bool = False,
):
    filt = TraversabilityFilter(w1, w2, w3, w_out, device=device, use_bias=use_bias)
    if device.startswith("cuda"):
        filt = filt.cuda().eval()
    else:
        filt = filt.cpu().eval()
    return filt

def unpack_rgb_from_float_cp(f_rgb: cp.ndarray) -> cp.ndarray:
    if f_rgb.dtype != cp.float32:
        raise TypeError("Input CuPy array must be of dtype float32.")
    
    H, W = f_rgb.shape
    bytes4 = f_rgb.view(cp.uint8).reshape(H, W, 4)
    r = bytes4[..., 2].copy()
    g = bytes4[..., 1].copy()
    b = bytes4[..., 0].copy()
    
    return cp.stack([r, g, b], axis=-1)

def pack_rgb_to_float_torch(rgb_uint8_tensor: torch.Tensor) -> torch.Tensor:
    if rgb_uint8_tensor.dim() != 3 or rgb_uint8_tensor.shape[2] != 3:
        raise ValueError("Input must be a HxWx3 PyTorch tensor.")
    if rgb_uint8_tensor.dtype != torch.uint8:
        raise TypeError("Input tensor dtype must be uint8.")
        
    H, W, _ = rgb_uint8_tensor.shape
    bytes4 = torch.zeros((H, W, 4), dtype=torch.uint8, device=rgb_uint8_tensor.device)
    bytes4[..., 0] = rgb_uint8_tensor[..., 2]  # Blue
    bytes4[..., 1] = rgb_uint8_tensor[..., 1]  # Green
    bytes4[..., 2] = rgb_uint8_tensor[..., 0]  # Red
    
    return bytes4.view(torch.float32).squeeze(-1)

def pack_rgb_to_float_np(rgb_uint8: np.ndarray) -> np.ndarray:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("Input must be a HxWx3 NumPy array.")
    if rgb_uint8.dtype != np.uint8:
        raise TypeError("Input array dtype must be uint8.")
        
    H, W, _ = rgb_uint8.shape
    bytes4 = np.zeros((H, W, 4), dtype=np.uint8)
    bytes4[..., 0] = rgb_uint8[..., 2]  # Blue
    bytes4[..., 1] = rgb_uint8[..., 1]  # Green
    bytes4[..., 2] = rgb_uint8[..., 0]  # Red
    return bytes4.view(np.float32).squeeze()

class InferenceHandler:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("Creating new InferenceHandler instance...")
            cls._instance = super(InferenceHandler, cls).__new__(cls)
            cls._instance._initialized = False
        else:
            print("Returning existing InferenceHandler instance...")
        return cls._instance
    
    def __init__(self, model_path: str, device: str = 'cuda', cell_n: int = 302, fps: float = 30.0,
                 rgb_weight: float = 0.50, geom_weight: float = 0.50,
                 use_trav_filter_bias: bool = False,
                 w1: Optional[np.ndarray] = None,
                 w2: Optional[np.ndarray] = None,
                 w3: Optional[np.ndarray] = None,
                 w_out: Optional[np.ndarray] = None):
        
        if self._initialized:
            return
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        ckpt = torch.load(model_path, map_location=self.device)
        self.args = ckpt['args'] if 'args' in ckpt else ckpt # Handle older checkpoints
        self.seq_len = self.args['seq_len']
        self.include_mask = self.args['include_mask']
        print("include_mask:", self.include_mask, ", seq_len:", self.seq_len)

        C_in = SEM_CHANNELS + 1 + (1 if self.include_mask else 0)
        C_out = SEM_CHANNELS + 1

        # self.model = UNetConvLSTM(
        #     in_ch=C_in, 
        #     base=self.args['base'], 
        #     out_ch=C_out
        # )
        self.model = CNNCorrectionNetSingle(
            in_ch_per_frame=C_in,
            base=self.args['base'],
            blocks_stage1=4,
            blocks_stage2=6,
            aspp_rates=(1, 2, 4),
            use_identity_correction=True,
            return_edit=False,
        )
        # self.model = DeepLabV3Plus(in_ch=C_in, out_ch=C_out)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()

        self.history = deque(maxlen=self.seq_len)
        print("Model loaded and ready for inference.")

        # Based on 300x300 crop + (1,1,1,1) padding
        buffer_size = 302 
        if cell_n != buffer_size:
             print(f"Warning: Plugin cell_n ({cell_n}) does not match hardcoded model output ({buffer_size}). This may fail if sizes are not compatible.")
             
        self.processed_elevation_buffer = cp.zeros((buffer_size, buffer_size), dtype=cp.float32)
        self.processed_rgb_buffer = cp.zeros((buffer_size, buffer_size), dtype=cp.float32)
        self.processed_rgb_traversability_buffer = cp.zeros((buffer_size, buffer_size), dtype=cp.float32)
        self.processed_geom_traversability_buffer = cp.zeros((buffer_size, buffer_size), dtype=cp.float32)
        self.processed_combined_cost_buffer = cp.zeros((buffer_size, buffer_size), dtype=cp.float32)

        self.last_update_time = -1.0
        self.fps = fps
        self.min_time_elapsed = 1.0 / self.fps
        
        self.rgb_weight = float(rgb_weight)
        self.geom_weight = float(geom_weight)
        
        if any(w is None for w in (w1, w2, w3, w_out)):
            w1d, w2d, w3d, w_outd = _default_weights()
            w1 = w1 if w1 is not None else w1d
            w2 = w2 if w2 is not None else w2d
            w3 = w3 if w3 is not None else w3d
            w_out = w_out if w_out is not None else w_outd
            
        self.traversability_filter = get_filter_torch(
            w1=w1, w2=w2, w3=w3, w_out=w_out, device=device, use_bias=use_trav_filter_bias
        )
        print(f"Traversability filter and all buffers initialized. Weights: RGB={self.rgb_weight}, Geom={self.geom_weight}")
        
        self._initialized = True

    def _preprocess_frame(self, rgb_layer_cp: cp.ndarray, elevation_layer_cp: cp.ndarray) -> torch.Tensor:
        y0, x0 = (rgb_layer_cp.shape[0] - 300) // 2, (rgb_layer_cp.shape[1] - 300) // 2
        rgb_cp = rgb_layer_cp[y0:y0+300, x0:x0+300]
        elev_cp = elevation_layer_cp[y0:y0+300, x0:x0+300]
        
        rgb_img_cp = unpack_rgb_from_float_cp(rgb_cp) # Now a (300, 300, 3) uint8 CuPy array

        # DEBUG
        # rgb_img_np = rgb_img_cp.get() 
        # bgr_img_np = cv2.cvtColor(rgb_img_np, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Input Frame", bgr_img_np)
        # cv2.waitKey(1) # Wait 1ms to allow the window to update
        # debug_print_old_class_counts(rgb_img_cp)
        
        # sem_onehot_cp = color28_to_onehot14(rgb_img_cp, dtype=cp.float32) # Stays on GPU
        sem_onehot_cp = color14_to_onehot14(rgb_img_cp, dtype=cp.float32) # Stays on GPU
        
        # 3. Process elevation (on GPU with CuPy)
        mask_cp = cp.isfinite(elev_cp).astype(cp.float32)
        elev_clean_cp = cp.nan_to_num(elev_cp, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 4. Concatenate all features (on GPU with CuPy)
        parts = [sem_onehot_cp, elev_clean_cp[..., None]]
        if self.include_mask:
            parts.append(mask_cp[..., None])
        
        frame_features_cp = cp.concatenate(parts, axis=-1) # Shape: [300, 300, C_in]
        
        # 5. Convert from CuPy to PyTorch tensor and permute to [C, H, W]
        # zero-copy operation between two GPU libraries.
        frame_tensor = torch.as_tensor(frame_features_cp, device=self.device)
        # frame_tensor = dlpack.from_dlpack(frame_features_cp.toDlpack()).to(self.device)
        return frame_tensor.permute(2, 0, 1) # [C_in, 300, 300]

    def _run_computation(self, rgb_layer_cp: cp.ndarray, elevation_layer_cp: cp.ndarray):
        frame_tensor = self._preprocess_frame(rgb_layer_cp, elevation_layer_cp) # [C, H, W]
        self.history.append(frame_tensor)
        if len(self.history) < self.seq_len:
            seq_list = [self.history[0]] * (self.seq_len - len(self.history)) + list(self.history)
        else:
            seq_list = list(self.history)
        
        input_tensor = torch.stack(seq_list, dim=0)  # Shape: [T, C, H, W]
        input_tensor = input_tensor.unsqueeze(0).to(self.device) # Shape: [B, T, C, H, W]

        with torch.no_grad():
            output_tensor = self.model(input_tensor).squeeze(0) # Shape: [C_out, H, W]

        pred_elev_tensor = output_tensor[SEM_CHANNELS].clamp(min=0)  # (300, 300)
        
        sem_logits_tensor = output_tensor[:SEM_CHANNELS]
        sem_indices_tensor = torch.argmax(sem_logits_tensor, dim=0)
        pred_rgb_uint8_tensor = onehot14_to_color(sem_indices_tensor) # (300, 300, 3)
        pred_rgb_float_tensor = pack_rgb_to_float_torch(pred_rgb_uint8_tensor) # (300, 300)
        
        padding = (1, 1, 1, 1)
        padded_rgb_tensor = F.pad(pred_rgb_float_tensor, pad=padding, mode='constant', value=0)
        padded_elev_tensor = F.pad(pred_elev_tensor, pad=padding, mode='constant', value=0)
        
        self.processed_rgb_buffer[...] = cp.asarray(padded_rgb_tensor)
        self.processed_elevation_buffer[...] = cp.asarray(padded_elev_tensor)
        
        inner_geom_trav_cu = self.traversability_filter.forward(self.processed_elevation_buffer)
        inner_geom_trav_cu = inner_geom_trav_cu.reshape(inner_geom_trav_cu.shape[-2], inner_geom_trav_cu.shape[-1])
        
        H, W = self.processed_elevation_buffer.shape
        geom_trav_full_cp = cp.ones((H, W), dtype=cp.float32)
        geom_trav_full_cp[3:-3, 3:-3] = inner_geom_trav_cu
        self.processed_geom_traversability_buffer[...] = geom_trav_full_cp

        padded_indices_tensor = F.pad(sem_indices_tensor, pad=padding, mode='constant', value=0)
        self.processed_rgb_traversability_buffer[...] = onehot14_to_traversability(cp.asarray(padded_indices_tensor))
        
        cp.multiply(self.processed_geom_traversability_buffer, self.geom_weight, 
                    out=self.processed_combined_cost_buffer)
        
        self.processed_combined_cost_buffer += (self.processed_rgb_traversability_buffer * self.rgb_weight)

    def run_inference(self, current_time: float, rgb_layer_cp: cp.ndarray, elevation_layer_cp: cp.ndarray):
        if (current_time - self.last_update_time) >= self.min_time_elapsed:
            self._run_computation(rgb_layer_cp, elevation_layer_cp)
            self.last_update_time = current_time