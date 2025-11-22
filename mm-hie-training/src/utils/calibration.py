from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn import calibration


def calibration_curve(y_true, y_probs, n_bins: int = 10) -> Dict[str, List[float]]:
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    prob_true, prob_pred = calibration.calibration_curve(y_true, y_probs, n_bins=n_bins)
    return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}


import torch
import torch.nn as nn


class TempScaler(nn.Module):
    """Simple temperature scaling module for logits.

    Useful for post-hoc calibration of (multi-label) logits via BCEWithLogitsLoss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single temperature parameter on logits/labels using BCEWithLogitsLoss.

    Args:
        logits: NumPy array of shape (N, C) with raw model logits.
        labels: NumPy array of shape (N, C) with multi-label targets (0/1).

    Returns:
        Scalar temperature as a Python float.
    """

    t_logits = torch.tensor(logits, dtype=torch.float32)
    t_labels = torch.tensor(labels, dtype=torch.float32)

    scaler = TempScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():  # type: ignore[return-type]
        optimizer.zero_grad()
        loss = criterion(scaler(t_logits), t_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.detach().cpu().numpy())
