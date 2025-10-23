#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scans a source directory for dataset shards, shuffles them, and physically
splits them into train, validation, and test sets in a new directory.

This script is designed to be safely re-run. It checks for shards that have
already been processed and only splits the newly found ones, preventing
data loss and overwrites. It also renames files to avoid name collisions.
"""
import os
import glob
import random
import shutil
import argparse
from tqdm import tqdm

def create_dataset_split(args):
    print(f"--- 1. Checking for existing shards in: {args.output_dir} ---")
    existing_files = set()
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(args.output_dir, split_name)
        if os.path.isdir(split_dir):
            for filepath in glob.glob(os.path.join(split_dir, '*.tar')):
                existing_files.add(os.path.basename(filepath))
    
    if existing_files:
        print(f"Found {len(existing_files)} shards already in output. They will be skipped.")

    print(f"\n--- 2. Searching for all shards in: {args.input_dir} ---")
    search_pattern = os.path.join(args.input_dir, '**', '*.tar')
    all_shards = glob.glob(search_pattern, recursive=True)

    if args.only_remapped:
        all_shards = [p for p in all_shards if 'remapped' in os.path.dirname(p)]
        print(f"Found {len(all_shards)} shards in '...remapped' subdirectories.")
    else:
        print(f"Found {len(all_shards)} total shards.")

    if not all_shards:
        print("Error: No shards found. Please check your --input-dir and --only-remapped flag.")
        return

    print("\n--- 3. Identifying new shards to be processed ---")
    new_shards = []
    for src_path in all_shards:
        relative_path = os.path.relpath(src_path, args.input_dir)
        new_filename = relative_path.replace(os.path.sep, '_')
        if new_filename not in existing_files:
            new_shards.append(src_path)

    if not new_shards:
        print("No new shards to process. The output directory is already up-to-date.")
        return
        
    print(f"Found {len(new_shards)} new shards to split and process.")

    print(f"\n--- 4. Shuffling new shards with seed: {args.split_seed} ---")
    random.Random(args.split_seed).shuffle(new_shards)

    if args.val_frac + args.test_frac >= 1.0:
        raise ValueError("The sum of val_frac and test_frac must be less than 1.0.")

    num_shards = len(new_shards)
    num_val = int(num_shards * args.val_frac)
    num_test = int(num_shards * args.test_frac)
    num_train = num_shards - num_val - num_test

    print(f"\nSplitting the {num_shards} new shards into:")
    print(f"  - Training:   {num_train} shards")
    print(f"  - Validation: {num_val} shards")
    print(f"  - Test:       {num_test} shards")
    print("-" * 20)

    train_files = new_shards[:num_train]
    val_files = new_shards[num_train : num_train + num_val]
    test_files = new_shards[num_train + num_val:]

    splits = { "train": train_files, "val": val_files, "test": test_files }

    print(f"\n--- 5. Copying new files ---")
    if args.dry_run:
        print("\n*** DRY RUN ENABLED: No files will be copied. ***\n")

    op_func = shutil.copy

    for split_name, files in splits.items():
        dest_dir = os.path.join(args.output_dir, split_name)
        
        if not args.dry_run:
            os.makedirs(dest_dir, exist_ok=True)

        print(f"Processing '{split_name}' set ({len(files)} files)...")
        for src_path in tqdm(files, desc=f"  -> {split_name}", ncols=100):
            relative_path = os.path.relpath(src_path, args.input_dir)
            new_filename = relative_path.replace(os.path.sep, '_')
            dest_path = os.path.join(dest_dir, new_filename)

            if args.dry_run:
                print(f"[DRY RUN] Copy '{src_path}' to '{dest_path}'")
            else:
                try:
                    op_func(src_path, dest_path)
                except Exception as e:
                    print(f"\nError processing {src_path}: {e}")
    
    print("\n--- 6. Data splitting process complete! ---")
    print(f"Output generated in: {args.output_dir}")


def build_argparser():
    p = argparse.ArgumentParser(description="Physically split dataset shards into train/val/test folders.")
    p.add_argument("--input-dir", required=True, help="Root directory containing all shards (e.g., .../carla3/data).")
    p.add_argument("--output-dir", required=True, help="Directory where train/val/test folders will be created.")
    p.add_argument("--val-frac", type=float, default=0.1, help="Fraction of shards for the validation set (default: 0.1).")
    p.add_argument("--test-frac", type=float, default=0.1, help="Fraction of shards for the test set (default: 0.1).")
    p.add_argument("--split-seed", type=int, default=357, help="Seed for the random shuffle to ensure reproducible splits.")
    p.add_argument("--only-remapped", action='store_true', help="If set, only use shards from directories containing 'remapped'.")
    p.add_argument("--dry-run", action='store_true', help="If set, print actions without actually moving or copying files.")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    create_dataset_split(args)

"""
python3 create_split.py \
--input-dir /media/slsecret/T7/carla3/data \
--output-dir /media/slsecret/T7/carla3/data_split357 \
--only-remapped
"""