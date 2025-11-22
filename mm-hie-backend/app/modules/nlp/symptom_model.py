from __future__ import annotations

from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from ...schemas import SymptomOutput, ConditionProb


class SymptomNLPModel:
    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.conditions = ["Pneumonia", "Sepsis", "Anemia", "COPD", "COVID-19"]

    @torch.inference_mode()
    def infer(self, text: str, top_n: int = 5) -> SymptomOutput:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**encoded)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

        probs = torch.softmax(torch.randn(len(self.conditions)), dim=0)
        top_n = min(top_n, len(self.conditions))
        top_indices = torch.topk(probs, k=top_n).indices.tolist()

        conditions = [
            ConditionProb(condition=self.conditions[i], prob=float(probs[i]))
            for i in top_indices
        ]

        return SymptomOutput(
            conditions=conditions,
            embedding=embedding.tolist(),
        )
