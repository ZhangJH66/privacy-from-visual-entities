#!/bin/bash
#
# Script to train ResNetPrivacy on a custom two-folder dataset.
#
# Expected dataset layout:
#   <DATA_DIR>/private/        <- privacy images   (label 0)
#   <DATA_DIR>/non_private/    <- non-privacy images (label 1)
#
# Usage:
#   bash scripts/run_custom_dataset.sh
#
# Before running, set DATA_DIR below (or export it as an environment variable)
# to the absolute path of your dataset root directory.
#
##############################################################################

DATA_DIR="${DATA_DIR:-/path/to/your/dataset}"

# Update configs/datasets.json so that data_prefix points to the parent
# of DATA_DIR and data_dir points to the dataset folder name.
# Alternatively, set data_dir to the full absolute path and leave
# data_prefix empty (the default).

python srcs/main.py \
    --config configs/custom_dataset_v1.0.json \
    --mode training \
    --dataset CustomDataset \
    --training_mode crossval \
    --fold_id 0
