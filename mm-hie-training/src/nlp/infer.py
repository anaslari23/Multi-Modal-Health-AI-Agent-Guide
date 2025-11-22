import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .train_nlp import PRETRAIN, NUM_LABELS, OUTPUT_DIR, MAX_LEN


def load_model(checkpoint_dir: Optional[str] = None):
    ckpt = checkpoint_dir or OUTPUT_DIR
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt,
        problem_type="multi_label_classification",
        num_labels=NUM_LABELS,
    )
    model.eval()
    return tokenizer, model


def predict(texts, top_k: int = 5, checkpoint_dir: Optional[str] = None):
    tokenizer, model = load_model(checkpoint_dir)
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits  # [B, num_labels]
        probs = torch.sigmoid(logits)

    results = []
    for i in range(probs.size(0)):
        p = probs[i]
        values, indices = torch.topk(p, k=min(top_k, p.size(0)))
        results.append(
            {
                "text": texts[i],
                "top_indices": indices.tolist(),
                "top_scores": [float(v) for v in values.tolist()],
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", nargs="*", help="Input text(s) to classify")
    parser.add_argument("--file", help="Optional path to a text file (one sample per line)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory (defaults to configs logging.checkpoint_dir)")
    args = parser.parse_args()

    texts: list[str] = []
    if args.text:
        texts.extend(args.text)

    if args.file:
        path = Path(args.file)
        if path.exists():
            texts.extend([line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

    if not texts:
        raise SystemExit("No input text provided. Use --text or --file.")

    results = predict(texts, top_k=args.top_k, checkpoint_dir=args.checkpoint)
    for r in results:
        print("===")
        print("TEXT:", r["text"])
        print("TOP_INDICES:", r["top_indices"])
        print("TOP_SCORES:", r["top_scores"])


if __name__ == "__main__":
    main()
