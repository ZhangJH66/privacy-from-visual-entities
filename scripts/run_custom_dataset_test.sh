#!/bin/bash
#
# Script to run inference on a custom two-folder dataset using
# PRE-TRAINED weights from an existing model (e.g. trained on
# PrivacyAlert or IPD) — no retraining needed.
#
# Supported image models whose weights can be reused directly:
#   RNP2FTP         (ResNet-50 fine-tuned with Places365 backbone)
#   ResNetPrivacy   (ResNet-50 fine-tuned with ImageNet backbone)
#
# Expected dataset layout:
#   <DATA_DIR>/private/        <- privacy images   (label 0)
#   <DATA_DIR>/non_private/    <- non-privacy images (label 1)
#
# Usage:
#   # Option A – point to a full path of the .pth checkpoint file:
#   MODEL_FILENAME=/path/to/best_acc_rnp2ftp-final.pth \
#       bash scripts/run_custom_dataset_test.sh
#
#   # Option B – point to the directory that contains the checkpoint:
#   MODEL_DIR=/path/to/trained_models/privacyalert \
#       bash scripts/run_custom_dataset_test.sh
#
##############################################################################

# --- configuration -----------------------------------------------------------
# Path to the .pth checkpoint file (takes priority over MODEL_DIR).
MODEL_FILENAME="${MODEL_FILENAME:-}"

# Directory that contains the checkpoint (used when MODEL_FILENAME is empty).
MODEL_DIR="${MODEL_DIR:-}"

# Config file must match the model architecture stored in the checkpoint.
# Use rnp2ftp_v1.0.json for RNP2FTP weights, or custom_dataset_v1.0.json
# for ResNetPrivacy weights.
CONFIG="${CONFIG:-configs/rnp2ftp_v1.0.json}"
# -----------------------------------------------------------------------------

EXTRA_ARGS=""
if [ -n "${MODEL_FILENAME}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --model_filename ${MODEL_FILENAME}"
elif [ -n "${MODEL_DIR}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --model_dir ${MODEL_DIR}"
fi

python srcs/main.py \
    --config "${CONFIG}" \
    --mode testing \
    --dataset CustomDataset \
    --split_mode test \
    --model_mode best \
    ${EXTRA_ARGS}
