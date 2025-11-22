import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments

from model_nlp import build_nlp_model_from_env
from nlp_dataset import (
    build_label_vocab,
    build_tokenized_datasets,
    load_jsonl_datasets,
)


CFG_PATH = os.environ.get("NLP_LARGE_CONFIG", "mm-hie-scale/training/scripts/configs/nlp_large.yaml")


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(pred):
    logits = pred.predictions
    y_true = pred.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred = (probs > 0.5).astype(int)

    metrics: Dict[str, float] = {}
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

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
        pass

    return metrics


class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.get("labels")
        if labels is not None:
            inputs["labels"] = labels.to(dtype=torch.float32)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


def extract_embeddings(model, dataset, out_dir: str, batch_size: int = 16) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            _, emb = model(input_ids=input_ids, attention_mask=attention_mask)
            all_emb.append(emb.cpu().numpy())
            all_labels.append(labels)

    emb_arr = np.concatenate(all_emb, axis=0)
    label_arr = np.concatenate(all_labels, axis=0)

    out_path = Path(out_dir) / "nlp_embeddings.npz"
    np.savez(out_path, emb=emb_arr, labels=label_arr)
    print(f"Saved NLP embeddings to {out_path}")


def main() -> None:
    cfg = load_config(CFG_PATH)

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    log_cfg = cfg.get("logging", {})

    pretrained = model_cfg.get("pretrained", "emilyalsentzer/Bio_ClinicalBERT")
    num_labels = int(model_cfg.get("num_labels", 50))
    max_len = int(data_cfg.get("max_length", 256))

    # Expose for model builder
    os.environ["NLP_LARGE_PRETRAIN"] = pretrained
    os.environ["NLP_LARGE_NUM_LABELS"] = str(num_labels)

    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    label_vocab_path = data_cfg.get("label_vocab", "data/nlp/labels.json")
    label_to_id = build_label_vocab(label_vocab_path)

    train_raw, val_raw = load_jsonl_datasets(data_cfg["train_path"], data_cfg["val_path"])
    train_ds, val_ds = build_tokenized_datasets(
        tokenizer,
        train_raw,
        val_raw,
        label_to_id,
        max_length=max_len,
    )

    model = build_nlp_model_from_env()

    output_dir = log_cfg.get("checkpoint_dir", "checkpoints/nlp")
    logging_dir = log_cfg.get("logging_dir", "logs/nlp_large")

    use_fp16 = bool(train_cfg.get("fp16", True)) and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(train_cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(train_cfg.get("batch_size", 8)),
        num_train_epochs=int(train_cfg.get("epochs", 5)),
        learning_rate=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        fp16=use_fp16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        logging_dir=logging_dir,
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=["tensorboard", "wandb"] if log_cfg.get("wandb_project") else ["tensorboard"],
    )

    if log_cfg.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", log_cfg["wandb_project"])

    trainer = MultiLabelTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Embedding extraction for fusion
    emb_dir = log_cfg.get("embedding_dir", "nlp_embeddings")
    extract_embeddings(model, train_ds, emb_dir, batch_size=int(train_cfg.get("embed_batch_size", 16)))


if __name__ == "__main__":
    main()
