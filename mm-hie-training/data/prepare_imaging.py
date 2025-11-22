import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "imaging"


def make_split(split: str, n_per_class: int = 4) -> None:
    split_dir = OUT_DIR / split
    normal_dir = split_dir / "normal"
    pneu_dir = split_dir / "pneumonia"
    normal_dir.mkdir(parents=True, exist_ok=True)
    pneu_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_per_class):
        # Normal: low-noise grayscale image
        arr_normal = np.random.normal(loc=0.5, scale=0.05, size=(224, 224)).clip(0, 1)
        img_normal = Image.fromarray((arr_normal * 255).astype("uint8"))
        img_normal = img_normal.convert("RGB")
        img_normal.save(normal_dir / f"normal_{i}.png")

        # Pneumonia: add a bright circular blob to simulate opacity
        arr_pneu = np.random.normal(loc=0.4, scale=0.08, size=(224, 224))
        yy, xx = np.ogrid[:224, :224]
        center_x, center_y = 112, 112
        radius = 40
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        arr_pneu[mask] += 0.5
        arr_pneu = arr_pneu.clip(0, 1)
        img_pneu = Image.fromarray((arr_pneu * 255).astype("uint8"))
        img_pneu = img_pneu.convert("RGB")
        img_pneu.save(pneu_dir / f"pneumonia_{i}.png")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_per_class", type=int, default=4)
    parser.add_argument("--val_per_class", type=int, default=2)
    args = parser.parse_args()

    print(f"Writing synthetic imaging data under {OUT_DIR} ...")
    make_split("train", n_per_class=args.train_per_class)
    make_split("val", n_per_class=args.val_per_class)
    print("Done. You can now run: python -m src.imaging.train_imaging --data_dir data/imaging")


if __name__ == "__main__":
    main()
