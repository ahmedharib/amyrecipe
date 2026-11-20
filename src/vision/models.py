"""Vision model architectures: ScratchCNN baseline and Transfer-Learned ResNet18 (Area A)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ScratchCNN(nn.Module):
    """4-block convolutional neural network trained from scratch with BatchNorm & Dropout."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_transfer_resnet18(
    num_classes: int = 10,
    dropout: float = 0.4,
    pretrained: bool = True,
) -> nn.Module:
    """Load ResNet18 pretrained on ImageNet and replace the classification head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_transfer_stage(model: nn.Module, stage: str) -> None:
    """Control trainable parameters for two-phase transfer learning.

    stage == 'feature_extract': freeze backbone, train fc head only.
    stage == 'fine_tune': unfreeze layer4 + fc head for end-to-end adaptation.
    """
    if stage == "feature_extract":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    elif stage == "fine_tune":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("layer4.") or name.startswith("fc.")
    else:
        raise ValueError(f"Unknown transfer stage: {stage}")
