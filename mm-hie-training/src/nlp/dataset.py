from __future__ import annotations

from typing import Tuple

import numpy as np
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


def load_jsonl_datasets(train_path: str, val_path: str) -> Tuple[Dataset, Dataset]:
    """Load train/val JSONL symptom data via HuggingFace datasets."""

    ds = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    return ds["train"], ds["validation"]


def build_tokenized_datasets(
    tokenizer: PreTrainedTokenizerBase,
    train_ds: Dataset,
    val_ds: Dataset,
    num_labels: int,
    max_length: int,
) -> Tuple[Dataset, Dataset]:
    """Tokenize text and multi-hot encode labels field for multi-label training."""

    def preprocess(batch):
        toks = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        arr = np.zeros((len(batch["labels"]), num_labels), dtype=int)
        for idx, labs in enumerate(batch["labels"]):
            arr[idx, labs] = 1
        toks["labels"] = arr.tolist()
        return toks

    train_tok = train_ds.map(preprocess, batched=True)
    val_tok = val_ds.map(preprocess, batched=True)
    return train_tok, val_tok

