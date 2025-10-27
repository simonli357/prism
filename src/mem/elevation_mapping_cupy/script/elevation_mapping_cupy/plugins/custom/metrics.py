#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def compute_elevation_metrics(map_a: np.ndarray, 
                              map_b: np.ndarray, 
                              mask: np.ndarray) -> dict:
    
    if mask.sum() == 0:
        return {'mae': np.nan, 'rmse': np.nan}
    
    mask = mask > 0.5
    
    diff = map_b[mask] - map_a[mask]
    
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    
    return {'mae': float(mae), 'rmse': float(rmse)}


def compute_miou(labels_a: np.ndarray, 
                 labels_b: np.ndarray, 
                 num_classes: int, 
                 ignore_label: int = -1) -> dict:
    
    gt_flat = labels_b.flatten()
    pred_flat = labels_a.flatten()
    
    valid_mask = (gt_flat != ignore_label)
    gt_flat_valid = gt_flat[valid_mask]
    pred_flat_valid = pred_flat[valid_mask]
    
    hist = np.bincount(
        num_classes * gt_flat_valid.astype(int) + pred_flat_valid.astype(int),
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    
    intersection = np.diag(hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)
    pr_sum = hist.sum(axis=0).astype(np.float64)
    union = gt_sum + pr_sum - intersection
    
    iou_per_class = np.full(num_classes, np.nan, dtype=np.float64)
    valid_union = (union > 0)
    iou_per_class[valid_union] = intersection[valid_union] / union[valid_union]
    
    valid_class_mask = np.ones(num_classes, dtype=bool)
    if 0 <= ignore_label < num_classes:
        valid_class_mask[ignore_label] = False
    
    miou = np.nanmean(iou_per_class[valid_class_mask])
    
    return {'miou': float(miou), 'iou_per_class': iou_per_class}

# --- BF-Score logic below is copied from plot_semantics.py ---

def seg_boundary_map(ids: np.ndarray) -> np.ndarray:
    b = np.zeros_like(ids, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            b |= np.roll(np.roll(ids, dy, axis=0), dx, axis=1) != ids
    return b

def dilate_mask(m: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return m
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tol + 1, 2 * tol + 1))
    return cv2.dilate(m.astype(np.uint8), kernel).astype(bool)

def compute_bf_score(labels_a: np.ndarray, 
                     labels_b: np.ndarray, 
                     tol: int = 2, 
                     ignore_label: int = -1) -> dict:
    
    pred_ids = labels_a
    gt_ids = labels_b
    
    pb = seg_boundary_map(pred_ids)
    gb = seg_boundary_map(gt_ids)
    
    pred_lab = (pred_ids != ignore_label)
    gt_lab = (gt_ids != ignore_label)
    
    pb &= pred_lab
    gb &= gt_lab
    
    gb_d = dilate_mask(gb, tol)
    pb_d = dilate_mask(pb, tol)
    
    tp_p = int((pb & gb_d).sum())
    tp_g = int((gb & pb_d).sum())
    
    pb_sum = int(pb.sum())
    gb_sum = int(gb.sum())

    prec = (tp_p / max(1, pb_sum)) if pb_sum > 0 else 0.0
    rec  = (tp_g / max(1, gb_sum)) if gb_sum > 0 else 0.0
    
    bf_score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    return {
        'bf_score': float(bf_score), 
        'precision': float(prec), 
        'recall': float(rec)
    }