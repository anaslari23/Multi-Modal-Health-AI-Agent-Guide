from __future__ import annotations

import torch
import torch.nn as nn


class VitalsTransformerModel(nn.Module):
    """Transformer encoder over vitals time-series.

    Input shape: [B, T, F] where F = number of vitals features.

    Returns:
        risk_logits: [B]  - vitals_risk score (before sigmoid)
        anomaly_logits: [B] - anomaly score (before sigmoid)
        embedding: [B, D] - pooled representation for fusion
    """

    def __init__(self, num_features: int, d_model: int = 128, nhead: int = 4, nlayers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.risk_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.anomaly_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor):  # x: [B, T, F]
        h = self.proj(x)  # [B, T, D]
        h = self.encoder(h)  # [B, T, D]
        h_pooled = self.pool(h.permute(0, 2, 1)).squeeze(-1)  # [B, D]

        risk_logits = self.risk_head(h_pooled).squeeze(-1)
        anomaly_logits = self.anomaly_head(h_pooled).squeeze(-1)
        return risk_logits, anomaly_logits, h_pooled


def contrastive_loss_stub(embeddings: torch.Tensor) -> torch.Tensor:
    """Placeholder contrastive objective.

    Currently returns 0 and can be replaced with NT-Xent or similar later.
    """

    return embeddings.new_zeros(())


def build_vitals_model(num_features: int, d_model: int = 128) -> VitalsTransformerModel:
    return VitalsTransformerModel(num_features=num_features, d_model=d_model)
