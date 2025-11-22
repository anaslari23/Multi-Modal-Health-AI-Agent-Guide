from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel


def load_tokenizer_and_classifier(pretrained: str, num_labels: int) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load ClinicalBERT tokenizer and multi-label classifier head."""

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained,
        problem_type="multi_label_classification",
        num_labels=num_labels,
    )
    return tokenizer, model


class ClinicalBERTEmbedder(nn.Module):
    """Embedding extractor that returns CLS embeddings for fusion (dim=768)."""

    def __init__(self, pretrained: str) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token embedding: [B, 768] for BERT-like models
        return outputs.last_hidden_state[:, 0, :]

