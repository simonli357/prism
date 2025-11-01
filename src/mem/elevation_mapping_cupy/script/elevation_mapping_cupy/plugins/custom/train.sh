#!/usr/bin/env bash
set -euo pipefail

EPOCHS=37
DEVICE="cuda"                  # "cuda" or "cpu"
SEQ_LEN=4
INCLUDE_MASK=true
BATCH_SIZE=8
BASE=64                        # base channels for UNet/CNN variants
LR=3e-4
LOSS_TYPE="focal"              # "ce" or "focal"
AUTO_CLASS_WEIGHTS=true        # only used when LOSS_TYPE=focal
WORKERS=8

SHARDS="/media/slsecret/T7/carla3/data_split357"
OUT_BASE="/media/slsecret/T7/carla3/runs/all357"   # per-model suffix will be appended
INFER_MAXCOUNT=10000
SAVE_LOGITS=false

# MODELS=("cnn" "unet" "unet_attn" "deeplabv3p")
MODELS=("unet" "cnn" "unet_correction" "deeplabv3p")
# MODELS=("unet_correction")
# --------------------------

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
  local MODEL_OUT_DIR="${OUT_BASE}${SUFFIX}"
  local TEST_SHARDS_DIR="${SHARDS}/test"
  local BEST_CKPT="${MODEL_OUT_DIR}/checkpoint_best.pt"
  local INFERENCE_OUT_DIR="${MODEL_OUT_DIR}/inference_on_test_set"


  echo
  echo "============================================================"
  echo " Model: ${MODEL}"
  echo " Output Dir: ${MODEL_OUT_DIR}"
  echo "============================================================"

  # --------------------------
  # TRAIN
  # --------------------------
  echo "--- Starting Training for ${MODEL} ---"
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
    # --resume "${MODEL_OUT_DIR}"

  echo "✔ Training complete for ${MODEL}."

  # --------------------------
  # TEST SET INFERENCE
  # --------------------------
  echo "--- Starting Test Set Inference for ${MODEL} ---"
  if [ ! -f "${BEST_CKPT}" ]; then
    echo "Warning: Best checkpoint not found at ${BEST_CKPT}. Skipping inference."
  else
    python3 run_inference.py \
      --checkpoint-path "${BEST_CKPT}" \
      --test-shards-dir "${TEST_SHARDS_DIR}" \
      --out-dir "${INFERENCE_OUT_DIR}" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}" \
      --workers "${WORKERS}"
      
    echo "✔ Inference complete for ${MODEL}. Results in ${INFERENCE_OUT_DIR}"
  fi

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