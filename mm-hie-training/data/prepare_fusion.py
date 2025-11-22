import os
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "fusion"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    num_samples = 128
    num_labels = 50

    records = []
    for i in range(num_samples):
        nlp = np.random.normal(size=(768,)).astype("float32")
        img = np.random.normal(size=(512,)).astype("float32")
        labs = np.random.normal(size=(50,)).astype("float32")
        vitals = np.random.normal(size=(128,)).astype("float32")

        # Multi-label indicator vector
        labels = (np.random.rand(num_labels) < 0.1).astype("float32")
        # Simple scalar risk as noisy sum of labels
        risk = float(labels.mean() + 0.1 * np.random.randn())

        records.append({
            "nlp": nlp,
            "img": img,
            "labs": labs,
            "vitals": vitals,
            "labels": labels,
            "risk": risk,
        })

    out_path = OUT_DIR / "train.npz"
    np.savez(out_path, records=np.array(records, dtype=object))
    print(f"Wrote synthetic fusion dataset to {out_path}")
    print("You can now run: python -m src.fusion.train_fusion --data data/fusion/train.npz")


if __name__ == "__main__":
    main()
