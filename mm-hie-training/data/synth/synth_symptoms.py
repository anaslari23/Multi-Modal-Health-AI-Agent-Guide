import json
import random
import argparse
from pathlib import Path
from typing import List, Dict


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "data" / "nlp"


SYMPTOM_SETS = [
    (["fever", "cough", "sore throat"], ["Flu", "Viral infection"]),
    (["fever", "retro-orbital pain", "rash", "joint pain"], ["Dengue"]),
    (["fever", "productive cough", "shortness of breath"], ["Pneumonia"]),
]

# Fixed label map so outputs are integer indices compatible with train_nlp.py
LABEL_MAP = {name: idx for idx, name in enumerate(["Flu", "Viral infection", "Dengue", "Pneumonia"])}


def _generate_samples(n: int) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for _ in range(n):
        symptoms, labels = random.choice(SYMPTOM_SETS)
        # Add some noise: random subset of the symptom list
        k = random.randint(1, len(symptoms))
        text = ", ".join(random.sample(symptoms, k=k))
        label_ids = [LABEL_MAP[l] for l in labels]
        samples.append({"text": text, "labels": label_ids})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--val_samples", type=int, default=400)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"

    train_samples = _generate_samples(args.train_samples)
    val_samples = _generate_samples(args.val_samples)

    with train_path.open("w", encoding="utf-8") as f:
        for item in train_samples:
            json.dump(item, f)
            f.write("\n")

    with val_path.open("w", encoding="utf-8") as f:
        for item in val_samples:
            json.dump(item, f)
            f.write("\n")

    print(f"Wrote {len(train_samples)} synthetic symptom samples to {train_path}")
    print(f"Wrote {len(val_samples)} synthetic symptom samples to {val_path}")


if __name__ == "__main__":
    main()
