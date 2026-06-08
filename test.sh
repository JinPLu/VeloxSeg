#!/usr/bin/env sh
set -eu

DATASET_NAME="${DATASET_NAME:-AutoPETII}"
MODEL_NAME="${MODEL_NAME:-VeloxSeg}"
TRAIN_CONFIG="${TRAIN_CONFIG:-./config/train_config_bs4.json}"
TEST_CONFIG="${TEST_CONFIG:-./config/test_config.json}"
CHECKPOINT_INDEX="${CHECKPOINT_INDEX:-val_best}"
TRAIN_DATE="${TRAIN_DATE:-09_12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_ID="${GPU_ID:-0}"
USE_HD95="${USE_HD95:-0}"

case "$DATASET_NAME" in
    AutoPETII)
        DEFAULT_MODEL_CONFIG="./config/models_config_autopetii.json"
        ;;
    Hecktor2022)
        DEFAULT_MODEL_CONFIG="./config/models_config_hecktor2022.json"
        ;;
    BraTS2021)
        DEFAULT_MODEL_CONFIG="./config/models_config_brats2021.json"
        ;;
    *)
        echo "Unsupported DATASET_NAME: $DATASET_NAME" >&2
        exit 2
        ;;
esac

MODEL_CONFIG="${MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"

python -u ./run_test.py \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --train_config "$TRAIN_CONFIG" \
    --model_config "$MODEL_CONFIG" \
    --test_config "$TEST_CONFIG" \
    --checkpoint_index "$CHECKPOINT_INDEX" \
    --num_workers "$NUM_WORKERS" \
    --gpu_id "$GPU_ID" \
    --train_date "$TRAIN_DATE" \
    --use_hd95 "$USE_HD95"
