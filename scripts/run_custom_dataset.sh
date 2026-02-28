#!/bin/bash
#
# Script to train ResNetPrivacy on a custom two-folder dataset.
#
# Expected dataset layout:
#   <DATA_DIR>/private/        <- privacy images   (label 0)
#   <DATA_DIR>/non_private/    <- non-privacy images (label 1)
#
# Usage:
#   DATA_DIR=/path/to/your/dataset bash scripts/run_custom_dataset.sh
#
# Before running, set DATA_DIR to the absolute path of your dataset root.
# You must also set data_dir in configs/datasets.json to match DATA_DIR,
# or set data_prefix to the parent directory and data_dir to the folder name.
#
##############################################################################

if [ -z "${DATA_DIR}" ]; then
    echo "Error: DATA_DIR is not set."
    echo "Usage: DATA_DIR=/path/to/your/dataset bash scripts/run_custom_dataset.sh"
    exit 1
fi

python srcs/main.py \
    --config configs/custom_dataset_v1.0.json \
    --mode training \
    --dataset CustomDataset \
    --training_mode crossval \
    --fold_id 0
