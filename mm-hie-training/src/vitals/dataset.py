from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class VitalsWindowDataset(Dataset):
    """Sliding-window dataset over episodic vitals stored in an .npz file.

    Expected arrays in the .npz file:
        x: shape (N, T, F)  - vitals time-series
        y: shape (N,)       - risk labels (float, 0/1)

    We create overlapping windows over the time axis for each episode and
    reuse the same label for all windows from that episode.
    """

    def __init__(
        self,
        data_path: str,
        window_size: int = 60,
        stride: int = 30,
    ) -> None:
        d = np.load(data_path)
        self.x = d["x"]  # (N, T, F)
        self.y = d.get("y", np.zeros((self.x.shape[0],), dtype="float32"))

        self.window_size = window_size
        self.stride = stride

        self.indices: List[Tuple[int, int]] = []
        for epi_idx in range(self.x.shape[0]):
            T = self.x[epi_idx].shape[0]
            if T <= 0:
                continue
            start = 0
            while start < T:
                end = start + self.window_size
                if end > T:
                    # use the last window of exact size if possible
                    if T >= self.window_size:
                        start = T - self.window_size
                        end = T
                    else:
                        # sequence shorter than window; just use the full sequence
                        start = 0
                        end = T
                    self.indices.append((epi_idx, start))
                    break
                self.indices.append((epi_idx, start))
                start += self.stride

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        epi_idx, start = self.indices[idx]
        T = self.x[epi_idx].shape[0]
        end = min(start + self.window_size, T)

        window = self.x[epi_idx, start:end]  # (t, F)
        window_t = torch.tensor(window, dtype=torch.float32)
        label = torch.tensor(self.y[epi_idx], dtype=torch.float32)
        return window_t, label
