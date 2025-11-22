from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


def load_efficientnet_b0(num_labels: int, pretrained: bool = True) -> nn.Module:
    """Load EfficientNet-B0 with a multi-label classification head.

    The final classifier is replaced with a dropout + linear layer producing
    `num_labels` logits for BCEWithLogitsLoss.
    """

    model = models.efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_labels),
    )
    return model


class EfficientNetB0Embedder(nn.Module):
    """Embedding extractor that returns a pooled feature vector for fusion."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        base = models.efficientnet_b0(pretrained=pretrained)
        # Keep everything up to the classifier input
        self.features = base.features
        self.avgpool = base.avgpool
        self.out_dim = base.classifier[1].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.avgpool(feats)
        pooled = pooled.flatten(1)  # [B, out_dim]
        return pooled

