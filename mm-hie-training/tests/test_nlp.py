from pathlib import Path
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Make sure the project src/ is on sys.path when running pytest
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nlp.train_nlp import PRETRAIN, NUM_LABELS


def test_nlp_forward_shape():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAIN,
        problem_type="multi_label_classification",
        num_labels=NUM_LABELS,
    )

    texts = [
        "Fever and cough for three days.",
        "Shortness of breath and chest pain.",
    ]
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**enc)
    logits = outputs.logits

    assert logits.shape[0] == len(texts)
    assert logits.shape[1] == NUM_LABELS
