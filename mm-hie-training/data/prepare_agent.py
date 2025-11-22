import os
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "agent"


def _write_split(path, n: int, prefix: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            # simple symptom / context to next-step instruction mapping
            example = {
                "input": f"Patient with chest pain episode {prefix}{i}. Symptoms: chest pain, dyspnea.",
                "output": "Order ECG, troponin, and chest X-ray. Assess hemodynamics and start monitoring.",
            }
            f.write(json.dumps(example) + "\n")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"

    _write_split(train_path, n=32, prefix="train_")
    _write_split(val_path, n=8, prefix="val_")

    print(f"Wrote synthetic agent train data to {train_path}")
    print(f"Wrote synthetic agent val data to {val_path}")
    print("You can now run: python -m src.agent.train_agent --train_file data/agent/train.jsonl --val_file data/agent/val.jsonl")


if __name__ == "__main__":
    main()
