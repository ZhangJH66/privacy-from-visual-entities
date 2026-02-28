#!/usr/bin/env python
#
# ResNet fine-tuned for binary privacy classification using ImageNet
# pretrained weights (no external Places365 dependency).
#
##############################################################################
# Authors:
# - Alessio Xompero, a.xompero@qmul.ac.uk
#
#  Created Date: 2025/02/28
# Modified Date: 2025/02/28
#
# MIT License
# Copyright (c) 2023-2025 GraphNEx
# -----------------------------------------------------------------------------

import torch.nn as nn
import torchvision.models as models

from srcs.utils import print_model_parameters

#####################################################################


class ResNetPrivacy(nn.Module):
    """ResNet backbone fine-tuned for binary image privacy classification.

    The backbone is initialised with ImageNet pretrained weights from
    torchvision, so no external model file is required.  The final fully
    connected layer is replaced with a ``num_out_classes``-way linear
    classifier and initialised with Xavier uniform initialisation.

    Supported backbone architectures (``net_params.backbone_arch``):
    ``"resnet18"``, ``"resnet34"``, ``"resnet50"`` (default),
    ``"resnet101"``, ``"resnet152"``.
    """

    def __init__(self, config):
        super(ResNetPrivacy, self).__init__()

        self.model_name = config["model_name"]
        self.n_out_classes = config["net_params"]["num_out_classes"]
        self.b_bce = config["params"]["use_bce"]

        backbone_arch = config["net_params"].get("backbone_arch", "resnet50")

        # Load ImageNet pretrained backbone
        backbone = models.__dict__[backbone_arch](pretrained=True)

        # Replace the classification head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, self.n_out_classes)
        nn.init.xavier_uniform_(backbone.fc.weight.data)
        if backbone.fc.bias is not None:
            backbone.fc.bias.data.zero_()

        # Allow all parameters to be optimised
        for param in backbone.parameters():
            param.requires_grad = True

        self.backbone = backbone

    # ------------------------------------------------------------------
    def print_number_parameters(self):
        print("\nResNet (ImageNet pretrained) parameters: ")
        print_model_parameters(self.backbone)

    def get_model_name(self):
        return self.model_name

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, img):
        return self.backbone(img)
