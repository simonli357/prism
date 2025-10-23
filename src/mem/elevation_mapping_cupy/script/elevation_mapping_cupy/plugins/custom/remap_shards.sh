#!/bin/bash

# This script automates the process of remapping WebDataset shards for all CARLA towns from town0 to town7.
# It iterates through each town, sets the appropriate input and output directories,
# and calls the remap_shards.py script with a set of default parameters.

# --- Configuration ---
# Base directory where all town data is stored.
BASE_DATA_DIR="/media/slsecret/T7/carla3/data"

# Suffix for the input directory containing the original shards.
INPUT_SUFFIX="gridmap_wds"

# Suffix for the output directory where remapped shards will be saved.
OUTPUT_SUFFIX="gridmap_wds_remapped"

# --- Main Loop ---
# This loop iterates through numbers 0, 1, 2, 3, 4, 5, 6, 7.
for i in {0..7}
do
  TOWN_NAME="town$i"
  IN_DIR="$BASE_DATA_DIR/$TOWN_NAME/$INPUT_SUFFIX"
  OUT_DIR="$BASE_DATA_DIR/$TOWN_NAME/$OUTPUT_SUFFIX"

  # Check if the input directory exists before processing to avoid errors.
  if [ -d "$IN_DIR" ]; then
    echo "=========================================================="
    echo "                Processing $TOWN_NAME"
    echo "=========================================================="
    echo "Input:  $IN_DIR"
    echo "Output: $OUT_DIR"
    echo ""

    # Call the Python script with the specified arguments.
    # Using the same defaults as your example usage.
    python3 remap_shards.py \
      --in_dir "$IN_DIR" \
      --out_dir "$OUT_DIR" \
      --recursive true \
      --onehot_dtype uint8 \
      --compression 3 \
      --keep_orig_png false \
      --overwrite false

    echo ""
    echo "Finished processing $TOWN_NAME."
    echo ""
  else
    echo "----------------------------------------------------------"
    echo "Skipping $TOWN_NAME: Input directory not found."
    echo "Checked path: $IN_DIR"
    echo "----------------------------------------------------------"
    echo ""
  fi
done

echo "All towns have been processed."
