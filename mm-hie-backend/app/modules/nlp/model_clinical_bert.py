from __future__ import annotations

from typing import List

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from ...schemas import SymptomOutput, ConditionProb


class ClinicalBERTSymptomModel:
    """Symptom NLP model based on Bio_ClinicalBERT.

    - Uses the "emilyalsentzer/Bio_ClinicalBERT" checkpoint.
    - Adds a multi-label classification head over a fixed disease label set.
    - Returns top-N conditions with probabilities and a 768-d embedding for fusion.
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Define a disease label set (30–80 labels); here we use 50 as an example.
        self.conditions: List[str] = [
            "Pneumonia",
            "Sepsis",
            "Anemia",
            "COPD",
            "COVID-19",
            "Asthma",
            "Heart failure",
            "Myocardial infarction",
            "Stroke",
            "Pulmonary embolism",
            "DVT",
            "Hypertension",
            "Diabetes",
            "CKD",
            "Liver cirrhosis",
            "Pancreatitis",
            "Appendicitis",
            "Cholecystitis",
            "UTI",
            "Pyelonephritis",
            "Meningitis",
            "Encephalitis",
            "Pneumothorax",
            "ARDS",
            "Bronchitis",
            "Sinusitis",
            "Otitis media",
            "Cellulitis",
            "Osteomyelitis",
            "Dementia",
            "Parkinson disease",
            "Epilepsy",
            "Depression",
            "Anxiety",
            "Bipolar disorder",
            "Schizophrenia",
            "Hyperthyroidism",
            "Hypothyroidism",
            "Obesity",
            "Malnutrition",
            "GERD",
            "IBD",
            "IBS",
            "Peptic ulcer disease",
            "Migraine",
            "Cluster headache",
            "Anaphylaxis",
            "Allergic rhinitis",
            "Dermatitis",
        ]
        self.num_labels = len(self.conditions)

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.activation = nn.Sigmoid()

    @torch.inference_mode()
    def infer(self, text: str, top_n: int = 5) -> SymptomOutput:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.backbone(**encoded)

        # Use mean pooled token embeddings as a 768-d sentence embedding for fusion.
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # (hidden_size,)

        logits = self.classifier(embedding)  # (num_labels,)
        probs = self.activation(logits)

        top_n = min(top_n, self.num_labels)
        top_indices = torch.topk(probs, k=top_n).indices.tolist()

        conditions = [
            ConditionProb(condition=self.conditions[i], prob=float(probs[i]))
            for i in top_indices
        ]

        return SymptomOutput(
            conditions=conditions,
            embedding=embedding.tolist(),
        )
