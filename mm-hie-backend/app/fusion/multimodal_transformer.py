from __future__ import annotations

from typing import Optional, List

import torch
from torch import nn

from ..schemas import SymptomOutput, LabResults, ImagingOutput, VitalsOutput


class MultimodalTransformerFusion(nn.Module):
    """Research-grade multimodal transformer fusion engine.

    Inputs:
      - NLP embedding:    768-d
      - Imaging embedding: 512-d
      - Labs feature vec:  50-d
      - Vitals embedding:  128-d

    Architecture:
      - Projects each modality to 256-d.
      - Applies multi-head self-attention across the 4 modality tokens.
      - Produces a fused 256-d vector.
      - Heads:
          * disease classifier (multi-label probabilities)
          * severity risk regressor (0–100)
    """

    def __init__(self, num_diseases: int = 50, d_model: int = 256, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.num_diseases = num_diseases
        self.d_model = d_model

        # Projections to a common 256-d space
        self.proj_nlp = nn.Linear(768, d_model)
        self.proj_img = nn.Linear(512, d_model)
        self.proj_labs = nn.Linear(50, d_model)
        self.proj_vitals = nn.Linear(128, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.disease_head = nn.Linear(d_model, num_diseases)
        self.risk_head = nn.Linear(d_model, 1)

        self.sigmoid = nn.Sigmoid()

    def _get_nlp_vec(self, nlp: Optional[SymptomOutput]) -> torch.Tensor:
        if nlp is None or not nlp.embedding:
            return torch.zeros(768)
        vec = torch.tensor(nlp.embedding, dtype=torch.float32)
        if vec.numel() != 768:
            vec = vec.flatten()
            if vec.numel() > 768:
                vec = vec[:768]
            else:
                pad = torch.zeros(768 - vec.numel())
                vec = torch.cat([vec, pad], dim=0)
        return vec

    def _get_imaging_vec(self, imaging: Optional[ImagingOutput]) -> torch.Tensor:
        if imaging is None or not imaging.embedding:
            return torch.zeros(512)
        vec = torch.tensor(imaging.embedding, dtype=torch.float32)
        if vec.numel() != 512:
            vec = vec.flatten()
            if vec.numel() > 512:
                vec = vec[:512]
            else:
                pad = torch.zeros(512 - vec.numel())
                vec = torch.cat([vec, pad], dim=0)
        return vec

    def _get_labs_vec(self, labs: Optional[LabResults]) -> torch.Tensor:
        # Simple fixed-size lab feature vector: take up to 50 lab values in sorted key order
        if labs is None or not labs.values:
            return torch.zeros(50)
        keys = sorted(labs.values.keys())
        vals: List[float] = []
        for k in keys:
            vals.append(labs.values[k].value)
            if len(vals) >= 50:
                break
        vec = torch.tensor(vals, dtype=torch.float32)
        if vec.numel() < 50:
            pad = torch.zeros(50 - vec.numel())
            vec = torch.cat([vec, pad], dim=0)
        return vec

    def _get_vitals_vec(self, vitals: Optional[VitalsOutput]) -> torch.Tensor:
        if vitals is None or not vitals.embedding:
            return torch.zeros(128)
        vec = torch.tensor(vitals.embedding, dtype=torch.float32)
        if vec.numel() != 128:
            vec = vec.flatten()
            if vec.numel() > 128:
                vec = vec[:128]
            else:
                pad = torch.zeros(128 - vec.numel())
                vec = torch.cat([vec, pad], dim=0)
        return vec

    @torch.inference_mode()
    def fuse(
        self,
        nlp: Optional[SymptomOutput],
        labs: Optional[LabResults],
        imaging: Optional[ImagingOutput],
        vitals: Optional[VitalsOutput],
    ) -> dict:
        """Fuse modalities and return fused vector, disease probabilities, and risk score.

        Returns dict with keys:
          - fused_vector: Tensor (256,)
          - disease_probs: Tensor (num_diseases,)
          - risk_score: float in [0, 100]
        """

        nlp_vec = self._get_nlp_vec(nlp)
        img_vec = self._get_imaging_vec(imaging)
        labs_vec = self._get_labs_vec(labs)
        vitals_vec = self._get_vitals_vec(vitals)

        nlp_proj = self.proj_nlp(nlp_vec.unsqueeze(0))      # (1, 256)
        img_proj = self.proj_img(img_vec.unsqueeze(0))      # (1, 256)
        labs_proj = self.proj_labs(labs_vec.unsqueeze(0))   # (1, 256)
        vitals_proj = self.proj_vitals(vitals_vec.unsqueeze(0))  # (1, 256)

        tokens = torch.stack(
            [nlp_proj, img_proj, labs_proj, vitals_proj], dim=1
        )  # (1, 4, 256)

        encoded = self.encoder(tokens)  # (1, 4, 256)
        fused = encoded.mean(dim=1).squeeze(0)  # (256,)

        disease_logits = self.disease_head(fused)  # (num_diseases,)
        disease_probs = self.sigmoid(disease_logits)

        risk_logit = self.risk_head(fused).squeeze(0)  # ()
        risk_score = float(self.sigmoid(risk_logit) * 100.0)

        return {
            "fused_vector": fused,
            "disease_probs": disease_probs,
            "risk_score": risk_score,
        }
