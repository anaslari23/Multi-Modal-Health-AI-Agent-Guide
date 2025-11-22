from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], ckpt_dir: str, name: str = "model.pt") -> Path:
    path = Path(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / name
    torch.save(state, out_path)
    return out_path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")
