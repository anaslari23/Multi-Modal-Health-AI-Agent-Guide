import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_jsonl_datasets(train_path: str, val_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = _load_jsonl(train_path)
    val = _load_jsonl(val_path)
    return train, val


def build_label_vocab(label_vocab_path: str) -> Dict[str, int]:
    """Load label->id mapping from JSON.

    Expected format: {"label_to_id": {"ICD10:I50.9": 0, ...}}
    """
    with open(label_vocab_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    label_to_id: Dict[str, int] = obj.get("label_to_id", obj)
    return label_to_id


def _example_to_text(example: Dict[str, Any]) -> str:
    """Best-effort flattening of notes + symptoms into one string."""
    pieces: List[str] = []
    symptoms = example.get("symptoms")
    notes = example.get("notes")
    text = example.get("text")

    if isinstance(symptoms, list):
        pieces.append("Symptoms: " + ", ".join(map(str, symptoms)))
    elif isinstance(symptoms, str):
        pieces.append("Symptoms: " + symptoms)

    if isinstance(notes, str):
        pieces.append(notes)
    elif isinstance(text, str):
        pieces.append(text)

    return " \n".join(pieces) if pieces else str(example)


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class NLPMultiLabelDataset(Dataset):
    """Tokenized multi-label disease classification dataset.

    Each JSONL record is expected to contain:
      - symptoms / notes / text fields (any subset)
      - labels: List[str] of canonical disease label keys
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        label_to_id: Dict[str, int],
        max_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

        texts: List[str] = []
        label_vectors: List[List[float]] = []

        num_labels = len(label_to_id)
        for rec in records:
            text = _example_to_text(rec)
            label_vec = [0.0] * num_labels
            for lab in rec.get("labels", []):
                idx = label_to_id.get(str(lab))
                if idx is not None and 0 <= idx < num_labels:
                    label_vec[idx] = 1.0
            texts.append(text)
            label_vectors.append(label_vec)

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        self.input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(label_vectors, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def build_tokenized_datasets(
    tokenizer: PreTrainedTokenizerBase,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    label_to_id: Dict[str, int],
    max_length: int,
) -> Tuple[NLPMultiLabelDataset, NLPMultiLabelDataset]:
    train_ds = NLPMultiLabelDataset(train_records, tokenizer, label_to_id, max_length=max_length)
    val_ds = NLPMultiLabelDataset(val_records, tokenizer, label_to_id, max_length=max_length)
    return train_ds, val_ds
