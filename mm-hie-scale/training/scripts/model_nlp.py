import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class NLPModelConfig:
    pretrained_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    num_labels: int = 50
    gradient_checkpointing: bool = True


class BioClinicalBertForMultiLabel(nn.Module):
    """Bio_ClinicalBERT backbone with a classification head and embedding output.

    The forward method is compatible with HuggingFace Trainer expectations:
      - inputs: input_ids, attention_mask, labels (optional)
      - returns: (loss, logits, embeddings) when labels are provided,
                 otherwise (logits, embeddings).
    """

    def __init__(self, cfg: NLPModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        base_config = AutoConfig.from_pretrained(
            cfg.pretrained_name,
            output_hidden_states=True,
        )
        base_config.num_labels = cfg.num_labels
        base_config.problem_type = "multi_label_classification"

        self.backbone = AutoModel.from_pretrained(cfg.pretrained_name, config=base_config)
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        hidden_size = base_config.hidden_size
        self.classifier = nn.Linear(hidden_size, cfg.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS embedding
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        cls_emb = last_hidden[:, 0, :]  # [B, H]
        logits = self.classifier(cls_emb)

        loss = None
        if labels is not None:
            labels = labels.to(dtype=torch.float32)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        if loss is not None:
            return loss, logits, cls_emb
        return logits, cls_emb


def build_nlp_model_from_env() -> BioClinicalBertForMultiLabel:
    pretrained = os.environ.get("NLP_LARGE_PRETRAIN", "emilyalsentzer/Bio_ClinicalBERT")
    num_labels = int(os.environ.get("NLP_LARGE_NUM_LABELS", "50"))
    cfg = NLPModelConfig(pretrained_name=pretrained, num_labels=num_labels, gradient_checkpointing=True)
    return BioClinicalBertForMultiLabel(cfg)
