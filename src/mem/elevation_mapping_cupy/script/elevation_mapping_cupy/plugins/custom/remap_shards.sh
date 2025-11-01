#!/bin/bash
BASE_DATA_DIR="/media/slsecret/T7/carla3/data"

INPUT_SUFFIX="gridmap_wds"

OUTPUT_SUFFIX="gridmap_wds_remapped"

for i in {0..7}
do
  for suffix in "" "b"; do
    TOWN_NAME="town$i$suffix"
    IN_DIR="$BASE_DATA_DIR/$TOWN_NAME/$INPUT_SUFFIX"
    OUT_DIR="$BASE_DATA_DIR/$TOWN_NAME/$OUTPUT_SUFFIX"

    if [ -d "$IN_DIR" ]; then
      echo "=========================================================="
      echo "                Processing $TOWN_NAME"
      echo "=========================================================="
      echo "Input:  $IN_DIR"
      echo "Output: $OUT_DIR"
      echo ""

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
      if [ -d "$BASE_DATA_DIR/$TOWN_NAME" ]; then
        echo "----------------------------------------------------------"
        echo "Skipping $TOWN_NAME: Input directory not found."
        echo "Checked path: $IN_DIR"
        echo "----------------------------------------------------------"
        echo ""
      fi
    fi
  done
done

echo "All towns have been processed."