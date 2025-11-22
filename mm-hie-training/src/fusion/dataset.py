from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    """Fusion dataset over precomputed modality embeddings.

    Expects an .npz file with a single array:
        records: object array of dicts with keys
            'nlp' (768,), 'img' (512,), 'labs' (50,), 'vitals' (128,),
            'labels' (num_labels,), 'risk' (scalar)
    """

    def __init__(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.records = data["records"]  # type: ignore[index]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        r = self.records[i]
        x = {k: torch.tensor(r[k]).float() for k in ("nlp", "img", "labs", "vitals")}
        labels = torch.tensor(r["labels"]).float()
        risk = torch.tensor(r["risk"]).float()
        return x, labels, risk
