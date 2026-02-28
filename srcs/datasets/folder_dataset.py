#!/usr/bin/env python
#
# Dataset class for loading privacy images from two folders.
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2025/02/28
# Modified Date: 2025/02/28
# -----------------------------------------------------------------------------

import os

import numpy as np
from PIL import Image
import torch.utils.data as data

from srcs.datasets.utils import set_img_transform

#############################################################################

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


class FolderPrivacyDataset(data.Dataset):
    """Dataset for binary privacy classification loaded from two image folders.

    Expected directory structure::

        data_dir/
            private/       <- privacy images   (label 0)
            non_private/   <- non-privacy images (label 1)

    The dataset is automatically split into train / val / test subsets using
    a deterministic random shuffle seeded with ``seed``.

    Partition modes
    ---------------
    * ``"final"`` :  train / test only
      (default 80 % / 20 %)
    * ``"crossval"`` or ``"original"`` : train / val / test
      (default 70 % / 10 % / 20 %)
    """

    def __init__(
        self,
        data_dir="",
        partition="final",
        split="train",
        num_classes=2,
        img_size=448,
        seed=789,
        train_ratio=0.8,
        val_ratio=0.1,
    ):
        """
        Arguments:
            - data_dir:     root directory that contains ``private/`` and
                            ``non_private/`` sub-folders.
            - partition:    split strategy – ``"final"`` or ``"crossval"`` /
                            ``"original"``.
            - split:        which subset to expose – ``"train"``, ``"val"``,
                            or ``"test"``.
            - num_classes:  number of output classes (default: 2).
            - img_size:     spatial size to resize images to (default: 448).
            - seed:         random seed for reproducible splitting.
            - train_ratio:  fraction of data used for training.
            - val_ratio:    fraction of data used for validation
                            (only used when partition != ``"final"``).
        """
        super(FolderPrivacyDataset, self).__init__()

        self.dataset_name = "CustomDataset"
        self.data_dir = data_dir
        self.partition = partition
        self.split_mode = split
        self.n_out_cls = num_classes
        self.im_size = img_size
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        assert partition in ["crossval", "final", "original"]
        if partition == "final":
            assert split in ["train", "test"]
        else:
            assert split in ["train", "val", "test"]

        self._load_data()

    # ------------------------------------------------------------------
    def _collect_images(self, folder, label):
        """Return a list of (abs_path, label) pairs for all images in folder."""
        items = []
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                "Expected image folder not found: {:s}\n"
                "CustomDataset requires both 'private/' and 'non_private/' "
                "subdirectories under the data directory: {:s}".format(
                    folder, self.data_dir
                )
            )
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                items.append((os.path.join(folder, fname), label))
        return items

    def _load_data(self):
        """Collect images, split and store the relevant subset."""
        private_dir = os.path.join(self.data_dir, "private")
        nonprivate_dir = os.path.join(self.data_dir, "non_private")

        private_items = self._collect_images(private_dir, 0)
        nonprivate_items = self._collect_images(nonprivate_dir, 1)

        all_items = private_items + nonprivate_items

        # Deterministic shuffle
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(all_items))
        rng.shuffle(indices)
        all_items = [all_items[i] for i in indices]

        n = len(all_items)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        if self.partition == "final":
            # train + test only; no validation set
            train_items = all_items[:n_train]
            test_items = all_items[n_train:]
            split_map = {"train": train_items, "test": test_items}
        else:
            # train + val + test
            train_items = all_items[:n_train]
            val_items = all_items[n_train : n_train + n_val]
            test_items = all_items[n_train + n_val :]
            split_map = {
                "train": train_items,
                "val": val_items,
                "test": test_items,
            }

        items = split_map[self.split_mode]

        self.imgs = [item[0] for item in items]
        self.labels = [item[1] for item in items]

        self._compute_weights()

    def _compute_weights(self):
        """Compute per-sample and per-class weights for imbalance handling."""
        cls_names = {0: "private", 1: "non_private"}
        cls_weights = np.zeros(self.n_out_cls)
        labels_arr = np.array(self.labels)
        l_weights = labels_arr.copy().astype(float)

        for l_idx in np.unique(labels_arr):
            n_imgs_cls = int(np.count_nonzero(labels_arr == l_idx))
            weight = 1.0 - n_imgs_cls / len(self.imgs)
            l_weights[labels_arr == l_idx] = weight
            cls_weights[l_idx] = weight
            print(
                "Number of {:s} images: {:d}".format(
                    cls_names[l_idx], n_imgs_cls
                )
            )

        self.weights = l_weights
        self.cls_weights = cls_weights

    # ------------------------------------------------------------------
    def get_class_weights(self):
        return self.cls_weights

    def get_name_low(self):
        return self.dataset_name.lower()

    def get_labels(self):
        return self.labels

    def get_data_dir(self):
        return self.data_dir

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        """Return (image_tensor, label, weight, filepath, original_size)."""
        full_path = self.imgs[index]
        img = Image.open(full_path).convert("RGB")
        w, h = img.size

        im_transform = set_img_transform(img_size=self.im_size)
        full_im = im_transform(img)

        target = self.labels[index]
        weight = self.weights[index]

        return full_im, target, weight, full_path, np.array([w, h])

    def __len__(self):
        return len(self.imgs)
