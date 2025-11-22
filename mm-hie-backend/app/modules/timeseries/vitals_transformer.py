from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from ...schemas import VitalsOutput
from .vitals_feature_extractor import build_vitals_tensor


@dataclass
class VitalsTransformerConfig:
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    window_size: int = 60


class VitalsTransformerEncoder(nn.Module):
    def __init__(self, config: VitalsTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(4, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.encoder(h)
        emb = h.mean(dim=1)
        logits = self.cls_head(emb)
        return emb, logits.squeeze(-1)


class VitalsTransformerModel:
    def __init__(self, config: Optional[VitalsTransformerConfig] = None) -> None:
        self.config = config or VitalsTransformerConfig()
        self.device = torch.device("cpu")
        self.model = VitalsTransformerEncoder(self.config).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def infer(
        self,
        heart_rate: List[float],
        spo2: List[float],
        temperature: List[float],
        resp_rate: List[float],
    ) -> VitalsOutput:
        x, stats = build_vitals_tensor(
            heart_rate=heart_rate,
            spo2=spo2,
            resp_rate=resp_rate,
            temperature=temperature,
            window_size=self.config.window_size,
            device=self.device,
        )

        emb, logits = self.model(x)
        risk = torch.sigmoid(logits).item()

        anomalies: List[str] = []
        if stats["hr_mean"] > 100 or stats["hr_max"] > 120:
            anomalies.append("tachycardia")
        if stats["spo2_min"] < 92:
            anomalies.append("oxygen_desaturation")
        if stats["temp_max"] > 38.0:
            anomalies.append("fever")
        if stats["rr_mean"] > 20:
            anomalies.append("tachypnea")

        return VitalsOutput(
            vitals_risk=float(risk),
            anomalies=anomalies,
            embedding=emb.squeeze(0).tolist(),
            heart_rate=list(heart_rate),
            spo2=list(spo2),
        )
