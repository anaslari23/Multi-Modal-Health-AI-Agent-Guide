import os
os.environ["WANDB_DISABLED"] = "true"

import yaml

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .dataset import load_jsonl_datasets, build_tokenized_datasets


CFG_PATH = os.environ.get("NLP_CONFIG", "configs/nlp.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_train_cfg = _cfg.get("train", {})
_data_cfg = _cfg.get("data", {})
_model_cfg = _cfg.get("model", {})
_log_cfg = _cfg.get("logging", {})


# Exposed constants (used by tests), allowing env overrides
PRETRAIN = os.environ.get("NLP_PRETRAIN", _model_cfg.get("pretrained", "emilyalsentzer/Bio_ClinicalBERT"))
NUM_LABELS = int(os.environ.get("NLP_NUM_LABELS", _model_cfg.get("num_labels", 50)))
MAX_LEN = int(os.environ.get("NLP_MAX_LEN", _data_cfg.get("max_length", 256)))
BATCH = int(os.environ.get("NLP_BATCH", _train_cfg.get("batch_size", 16)))
OUTPUT_DIR = os.environ.get("NLP_CHECKPOINT", _log_cfg.get("checkpoint_dir", "checkpoints/nlp"))
TRAIN_PATH = os.environ.get("NLP_TRAIN_PATH", _data_cfg.get("train_path", "data/nlp/train.jsonl"))
VAL_PATH = os.environ.get("NLP_VAL_PATH", _data_cfg.get("val_path", "data/nlp/val.jsonl"))
EPOCHS = int(os.environ.get("NLP_EPOCHS", _train_cfg.get("epochs", 6)))
LR = float(os.environ.get("NLP_LR", _train_cfg.get("lr", 2e-5)))
WEIGHT_DECAY = float(os.environ.get("NLP_WEIGHT_DECAY", _train_cfg.get("weight_decay", 0.01)))
WARMUP_STEPS = int(os.environ.get("NLP_WARMUP_STEPS", _train_cfg.get("warmup_steps", 0)))


def compute_metrics(pred):
    logits = pred.predictions
    y_true = pred.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred = (probs > 0.5).astype(int)

    metrics = {}
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-label AUC and macro AUC (best-effort; skip labels without both classes)
    aucs = []
    try:
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) < 2:
                continue
            auc = roc_auc_score(y_true[:, i], probs[:, i])
            metrics[f"auc_label_{i}"] = float(auc)
            aucs.append(auc)
        if aucs:
            metrics["auc_macro"] = float(np.mean(aucs))
    except Exception:
        # In small synthetic datasets some labels may be degenerate; ignore AUC failures.
        pass

    return metrics


class MultiLabelTrainer(Trainer):
    """Custom Trainer that ensures labels are float for BCEWithLogitsLoss.

    Some Transformers versions expect float labels for multi-label tasks, and
    passing integer labels can cause a dtype cast error. This wrapper casts
    labels to float before forwarding to the model.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.get("labels")
        if labels is not None:
            inputs["labels"] = labels.to(dtype=torch.float32)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAIN,
        problem_type="multi_label_classification",
        num_labels=NUM_LABELS,
    )

    # Load and tokenize datasets via helpers
    train_raw, val_raw = load_jsonl_datasets(TRAIN_PATH, VAL_PATH)
    train_dataset, val_dataset = build_tokenized_datasets(
        tokenizer,
        train_raw,
        val_raw,
        num_labels=NUM_LABELS,
        max_length=MAX_LEN,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir="logs/nlp",
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
