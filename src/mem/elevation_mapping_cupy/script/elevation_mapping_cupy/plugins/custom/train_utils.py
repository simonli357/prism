#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, io, json, argparse
from collections import defaultdict, deque
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='numpy')

import time, platform, sys, json, hashlib
from datetime import datetime
try:
    from .unet_conv_lstm import UNetConvLSTM
    from .unet_attention import AttentionUNetConvLSTM
    from .shard_utils import resolve_shards, SeqFromWDS
    from .deeplabv3plus import DeepLabV3Plus
    from .cnn import CNNCorrectionNetSingle
    from .plotter import _read_training_log_csv, _plot_metric, _save_training_plots
    from .evaluate3 import run_inference
except ImportError:
    from unet_conv_lstm import UNetConvLSTM
    from unet_attention import AttentionUNetConvLSTM
    from shard_utils import resolve_shards, SeqFromWDS
    from deeplabv3plus import DeepLabV3Plus
    from cnn import CNNCorrectionNetSingle
    from plotter import _read_training_log_csv, _plot_metric, _save_training_plots
    from evaluate3 import run_inference

try:
    from .color_mapping import color28_to_onehot14, color28_to_color14, onehot14_to_color, SEM_CHANNELS, NEW_CLASSES
except ImportError:
    from color_mapping import color28_to_onehot14, color28_to_color14, onehot14_to_color, SEM_CHANNELS, NEW_CLASSES

import random
def set_seed(seed=357):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def latest_epoch_ckpt(path_dir, prefix="checkpoint_epoch_", ext=".pt"):
    if not os.path.isdir(path_dir): 
        return None
    cks = [p for p in os.listdir(path_dir) if p.startswith(prefix) and p.endswith(ext)]
    if not cks:
        return None
    cks.sort()  # names are zero-padded, so lexical sort works
    return os.path.join(path_dir, cks[-1])
def to_serializable(obj):
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "state_dict"):
        return str(obj.__class__.__name__)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def save_json(path, data_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=2, default=to_serializable)
    
def model_num_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}

def env_info():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_is_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "devices": [
            {"idx": i, "name": torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())
        ],
    }

def sha1_of_file(path, block_size=1<<20):
    if not os.path.exists(path): return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b: break
            h.update(b)
    return h.hexdigest()
  
MODEL_SUFFIX = {
    "cnn": "_cnn",
    "unet": "_unet",
    "unet_attn": "_unet_attention",
    "deeplabv3p": "_deeplabv3p",
}

def resolve_out_dir(base_out: str, model_name: str) -> str:
    suffix = MODEL_SUFFIX.get(model_name, f"_{model_name}")
    return base_out.rstrip("/") + suffix