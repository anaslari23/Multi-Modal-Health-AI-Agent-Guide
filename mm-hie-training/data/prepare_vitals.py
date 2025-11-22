import os
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "vitals"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Simple synthetic vitals: HR, SpO2 over T timesteps
    N = 256  # number of sequences
    T = 48   # timesteps
    F = 2    # features: [HR, SpO2]

    # baseline normal vitals
    hr_base = np.random.normal(loc=80.0, scale=10.0, size=(N, T, 1))
    spo2_base = np.random.normal(loc=97.0, scale=1.0, size=(N, T, 1))

    x = np.concatenate([hr_base, spo2_base], axis=-1).astype("float32")

    # Binary labels: mark sequences with higher HR / lower SpO2 as positive
    risk = ((x[..., 0].mean(axis=1) > 90.0) | (x[..., 1].mean(axis=1) < 95.0)).astype("float32")

    out_path = OUT_DIR / "train.npz"
    np.savez(out_path, x=x, y=risk)
    print(f"Wrote synthetic vitals dataset to {out_path}")
    print("You can now run: python -m src.vitals.train_vitals --data data/vitals/train.npz")


if __name__ == "__main__":
    main()
