from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MultiLabelImageDataset(Dataset):
    """Imaging dataset for multi-label classification.

    Expects a list of image paths and corresponding multi-hot label vectors.
    Optionally applies a transform (e.g., albumentations) that takes/returns
    dicts with "image" key.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        transform=None,
    ) -> None:
        assert len(image_paths) == labels.shape[0]
        self.image_paths = [str(p) for p in image_paths]
        self.labels = labels.astype("float32")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(image=np.array(img))
            img_tensor = sample["image"]
        else:
            # Fallback: simple ToTensor-like conversion
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        label = torch.from_numpy(self.labels[idx])
        return img_tensor, label

