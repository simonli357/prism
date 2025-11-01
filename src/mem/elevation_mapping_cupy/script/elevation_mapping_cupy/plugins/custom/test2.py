#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import torch.multiprocessing as mp

try:
    from .train2 import build_model, _collate, evaluate
    from .shard_utils import resolve_shards, SeqFromWDS
    from .evaluate3 import run_inference
    from .train_utils import set_seed, save_json, to_serializable
    from .color_mapping import SEM_CHANNELS
except ImportError:
    from train2 import build_model, _collate, evaluate
    from shard_utils import resolve_shards, SeqFromWDS
    from evaluate3 import run_inference
    from train_utils import set_seed, save_json, to_serializable
    from color_mapping import SEM_CHANNELS

warnings.filterwarnings("ignore", category=UserWarning, module='numpy')

def main(args):
    set_seed(42)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    UNLABELED_ID = 0

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")
    print(f"Using device: {device}")

    if not os.path.isfile(args.checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint_path}")
        return

    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=device)

    if 'args' not in ckpt or 'model' not in ckpt:
        print("[ERROR] Checkpoint is invalid. Must contain 'args' (dict) and 'model' (state_dict).")
        return

    train_args = argparse.Namespace(**ckpt['args'])
    
    print(f"--- Model Config (from checkpoint) ---")
    print(f"  Model type:     {train_args.model}")
    print(f"  Base channels:  {train_args.base}")
    print(f"  Sequence len:   {train_args.seq_len}")
    print(f"  Include mask:   {train_args.include_mask}")
    print(f"----------------------------------------")

    C_in  = SEM_CHANNELS + 1 + (1 if train_args.include_mask else 0)
    C_out = SEM_CHANNELS + 1
    model = build_model(train_args, C_in, C_out, device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("Model loaded successfully.")

    print(f"Resolving test shards from: {args.test_shards_dir}")
    test_shards = resolve_shards(args.test_shards_dir, recursive=False, only_remapped=False)
    if not test_shards:
        print("[ERROR] No test shards found in directory.")
        return
    print(f"Found {len(test_shards)} test shards.")

    ds_test = SeqFromWDS(
        shard_paths=test_shards,
        seq_len=train_args.seq_len,
        shuffle_shards=False,
        include_mask=train_args.include_mask
    )
    
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=(args.workers > 0),
        prefetch_factor=(4 if args.workers > 0 else None),
    )

    print("\n--- Calculating Test Set Metrics ---")
    test_metrics = evaluate(model, test_loader, device, unlabeled_id=UNLABELED_ID)
    
    print("\n" + "="*40)
    print("          FINAL TEST SET METRICS")
    print("="*40)
    print(f" >> mIoU:      {test_metrics['miou']:.4f}")
    print(f" >> Accuracy:  {test_metrics['pix_acc']:.4f}")
    print(f" >> Elev. MAE: {test_metrics['elev_mae']:.4f}")
    print(f" >> Elev. RMSE:{test_metrics['elev_rmse']:.4f}")
    print(f" >> BF Score:  {test_metrics['bf_score']:.4f}")
    print("="*40 + "\n")

    metrics_path = os.path.join(args.out_dir, "test_metrics.json")
    save_json(metrics_path, to_serializable(test_metrics))
    print(f"Test metrics saved to: {metrics_path}")

    print("\n--- Starting Inference (Saving Results) ---")
    print(f"Saving outputs to: {args.out_dir}")
    
    try:
        run_inference(
            checkpoint_path=args.checkpoint_path,
            shards_list=test_shards,
            output_dir=args.out_dir,
            device=str(device),
            seq_len=train_args.seq_len,
            model_name=train_args.model,
            sem_channels=SEM_CHANNELS,
            include_mask=train_args.include_mask
        )
        print(f"--- Inference complete. Results saved in: {args.out_dir} ---")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during the inference saving step: {e}")
        import traceback
        traceback.print_exc()

def build_argparser():
    p = argparse.ArgumentParser(description="Run inference on a test set with a trained model.")
    p.add_argument("--checkpoint-path", required=True,
                   help="Path to the trained model checkpoint (.pt file).")
    p.add_argument("--test-shards-dir", required=True,
                   help="Directory containing the 'test' .tar shards.")
    p.add_argument("--out-dir", required=True,
                   help="Directory to save inference results and metrics.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for inference (adjust based on VRAM).")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of workers for data loading.")
    return p

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("--- Multiprocessing start method set to 'spawn' ---")
    except RuntimeError:
        pass

    args = build_argparser().parse_args()
    main(args)
    
"""
python3 test2.py \
    --checkpoint-path /media/slsecret/T7/carla3/runs/all357_cnn/checkpoint_best.pt \
    --test-shards-dir /media/slsecret/T7/carla3/data_split357/test \
    --out-dir /media/slsecret/T7/carla3/runs/all357_cnn/inference_on_test_set \
    --batch-size 8 \
    --workers 4 \
    --device cuda
"""