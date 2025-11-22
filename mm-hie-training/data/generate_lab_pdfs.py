from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
LABS_DIR = BASE_DIR / "data" / "labs"


def generate_lab_pdfs(csv_path: str, out_dir: str, limit: int = 50) -> None:
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= limit:
            break
        pid = row.get("patient_id", f"P{i}")

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis("off")

        lines = [f"Patient: {pid}"]
        for col in df.columns:
            if col == "patient_id":
                continue
            lines.append(f"{col}: {row[col]}")

        text = "\n".join(lines)
        ax.text(0.01, 0.99, text, va="top", fontsize=8)

        img_file = out_path / f"lab_{pid}.png"
        fig.savefig(img_file, bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"Wrote up to {limit} lab PDF-like PNGs under {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(LABS_DIR / "labs.csv"))
    parser.add_argument("--outdir", default=str(LABS_DIR / "pdfs"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    generate_lab_pdfs(args.csv, args.outdir, limit=args.limit)


if __name__ == "__main__":
    main()
