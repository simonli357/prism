#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# CONFIG - EDIT THESE
# --------------------------
EPOCHS=30
DEVICE="cuda"                  # "cuda" or "cpu"
SEQ_LEN=4
INCLUDE_MASK=true
BATCH_SIZE=8
BASE=64                        # base channels for UNet/CNN variants
LR=3e-4
LOSS_TYPE="focal"              # "ce" or "focal"
AUTO_CLASS_WEIGHTS=true        # only used when LOSS_TYPE=focal
WORKERS=4

# Data & IO
SHARDS="/media/slsecret/T7/carla3/data_split357"
OUT_BASE="/media/slsecret/T7/carla3/runs/all357"   # per-model suffix will be appended
INFER_MAXCOUNT=10000
SAVE_LOGITS=false

# Which models to run
# MODELS=("cnn" "unet" "unet_attn" "deeplabv3p")
# MODELS=("unet" "cnn" "deeplabv3p" "unet_correction")
MODELS=("unet_correction")
# --------------------------

# Small helper: map model -> suffix (must match your train2.py resolve_out_dir)
suffix_for_model() {
  case "$1" in
    cnn)          echo "_cnn" ;;
    unet)         echo "_unet" ;;
    unet_attn)    echo "_unet_attention" ;;
    deeplabv3p)   echo "_deeplabv3p" ;;
    unet_correction)   echo "_unet_correction" ;;
    *)            echo "_${1}" ;;
  esac
}

train_and_eval() {
  local MODEL="$1"
  local SUFFIX
  SUFFIX="$(suffix_for_model "$MODEL")"

  echo
  echo "============================================================"
  echo " Model: ${MODEL}"
  echo "============================================================"

  # --------------------------
  # TRAIN
  # --------------------------
  python3 train2.py \
    --model "${MODEL}" \
    --shards "${SHARDS}" \
    --out "${OUT_BASE}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --seq-len "${SEQ_LEN}" \
    --base "${BASE}" \
    --lr "${LR}" \
    --include-mask "${INCLUDE_MASK}" \
    --loss-type "${LOSS_TYPE}" \
    --auto-class-weights "${AUTO_CLASS_WEIGHTS}" \
    # --resume /media/slsecret/T7/carla3/runs/all357${SUFFIX}

  echo "✔ Completed model: ${MODEL}"
}

main() {
  for MODEL in "${MODELS[@]}"; do
    train_and_eval "${MODEL}"
  done
  echo
  echo "All models completed."
}

main "$@"
