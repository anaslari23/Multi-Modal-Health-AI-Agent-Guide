from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalTransformer(nn.Module):
    """Cross-modal transformer over precomputed modality embeddings.

    Modalities: nlp, img, labs, vitals.
    Each modality is projected to a shared dimension and encoded with a
    Transformer encoder.

    Forward returns:
        logits: [B, L]   - multi-label diagnosis scores (before sigmoid)
        risk:   [B]      - scalar risk regression (before any activation)
        fused:  [B, D]   - fused representation for downstream use
    """

    def __init__(
        self,
        dims=(768, 512, 50, 128),
        proj_dim: int = 256,
        nhead: int = 4,
        nlayers: int = 2,
        num_labels: int = 50,
    ) -> None:
        super().__init__()
        self.proj_nlp = nn.Linear(dims[0], proj_dim)
        self.proj_img = nn.Linear(dims[1], proj_dim)
        self.proj_labs = nn.Linear(dims[2], proj_dim)
        self.proj_vitals = nn.Linear(dims[3], proj_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels),
        )
        self.risk_head = nn.Sequential(nn.Linear(proj_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(
        self,
        nlp: torch.Tensor,
        img: torch.Tensor,
        labs: torch.Tensor,
        vitals: torch.Tensor,
    ):
        # project each modality and build a length-4 sequence (seq_len, batch, dim)
        a = self.proj_nlp(nlp).unsqueeze(0)
        b = self.proj_img(img).unsqueeze(0)
        c = self.proj_labs(labs).unsqueeze(0)
        d = self.proj_vitals(vitals).unsqueeze(0)
        x = torch.cat([a, b, c, d], dim=0)

        out = self.encoder(x)
        pooled = out.mean(dim=0)

        logits = self.classifier(pooled)
        risk = self.risk_head(pooled).squeeze(-1)
        return logits, risk, pooled
