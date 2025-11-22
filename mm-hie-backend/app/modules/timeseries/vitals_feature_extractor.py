from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch


def _to_array(series: List[float], fallback: float) -> np.ndarray:
    if not series:
        return np.array([fallback], dtype=np.float32)
    return np.asarray(series, dtype=np.float32)


def build_vitals_tensor(
    heart_rate: List[float],
    spo2: List[float],
    resp_rate: List[float],
    temperature: List[float],
    window_size: int = 60,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, dict]:
    """Convert raw vitals into a normalized time-series tensor.

    Returns
    -------
    seq : torch.Tensor
        Shape [1, T, 4] with channels [HR, SpO2, RR, Temp].
    stats : dict
        Simple statistics for downstream anomaly tagging.
    """

    hr = _to_array(heart_rate, 80.0)
    s2 = _to_array(spo2, 97.0)
    rr = _to_array(resp_rate, 16.0)
    tp = _to_array(temperature, 37.0)

    length = int(max(hr.size, s2.size, rr.size, tp.size))
    hr = _pad_or_truncate(hr, length)
    s2 = _pad_or_truncate(s2, length)
    rr = _pad_or_truncate(rr, length)
    tp = _pad_or_truncate(tp, length)

    hr_norm = (hr - 80.0) / 40.0
    s2_norm = (s2 - 97.0) / 5.0
    rr_norm = (rr - 16.0) / 8.0
    tp_norm = (tp - 37.0) / 1.5

    seq = np.stack([hr_norm, s2_norm, rr_norm, tp_norm], axis=-1)

    if length > window_size:
        seq = seq[-window_size:]
    else:
        pad_len = window_size - length
        if pad_len > 0:
            pad_block = np.repeat(seq[:1], pad_len, axis=0)
            seq = np.concatenate([pad_block, seq], axis=0)

    stats = {
        "hr_mean": float(hr.mean()),
        "hr_max": float(hr.max()),
        "spo2_min": float(s2.min()),
        "temp_max": float(tp.max()),
        "rr_mean": float(rr.mean()),
    }

    tensor = torch.from_numpy(seq).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor, stats


def _pad_or_truncate(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.size == length:
        return arr
    if arr.size > length:
        return arr[-length:]
    pad_len = length - arr.size
    pad_val = float(arr[-1])
    pad_block = np.full((pad_len,), pad_val, dtype=arr.dtype)
    return np.concatenate([pad_block, arr], axis=0)
