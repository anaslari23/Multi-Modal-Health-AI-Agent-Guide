from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from .calibration import calibration_curve, temperature_scale


def isotonic_regression(y_probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Fit isotonic regression to calibrate probabilities.

    Args:
        y_probs: Predicted probabilities (N,).
        y_true: True binary labels (N,).

    Returns:
        Calibrated probabilities (N,).
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    y_probs_calibrated = iso.fit_transform(y_probs, y_true)
    return y_probs_calibrated


def evaluate_module(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    module_name: str,
    n_bins: int = 10,
    apply_temperature: bool = False,
    apply_isotonic: bool = False,
) -> Dict[str, object]:
    """Evaluate calibration and performance for a single module.

    Args:
        y_true: True multi-label targets (N, C) or binary labels (N,).
        y_probs: Predicted probabilities (N, C) or (N,).
        module_name: Name of the module for reporting.
        n_bins: Number of bins for calibration curve.
        apply_temperature: Whether to apply temperature scaling.
        apply_isotonic: Whether to apply isotonic regression.

    Returns:
        Dictionary with metrics and calibration data.
    """
    # Ensure binary for Brier score and calibration curve
    if y_true.ndim > 1:
        # Multi-label: average over labels
        brier = np.mean(
            [
                brier_score_loss(y_true[:, i], y_probs[:, i])
                for i in range(y_true.shape[1])
            ]
        )
        # Use first label for calibration curve (or average)
        y_true_bin = y_true[:, 0]
        y_probs_bin = y_probs[:, 0]
    else:
        brier = brier_score_loss(y_true, y_probs)
        y_true_bin = y_true
        y_probs_bin = y_probs

    cal_curve = calibration_curve(y_true_bin, y_probs_bin, n_bins=n_bins)

    result: Dict[str, object] = {
        "module": module_name,
        "brier_score": float(brier),
        "calibration_curve": cal_curve,
    }

    # Temperature scaling (requires logits)
    if apply_temperature:
        # For demo: generate fake logits from probs using logit transform
        eps = 1e-12
        logits = np.log(y_probs_bin / (1 - y_probs_bin + eps) + eps)
        temp = temperature_scale(logits.reshape(-1, 1), y_true_bin.reshape(-1, 1))
        result["temperature"] = temp

    # Isotonic regression
    if apply_isotonic:
        y_probs_iso = isotonic_regression(y_probs_bin, y_true_bin)
        result["isotonic_calibrated_probs"] = y_probs_iso.tolist()

    return result


def evaluate_all_modules(
    nlp_probs: np.ndarray,
    nlp_labels: np.ndarray,
    imaging_probs: np.ndarray,
    imaging_labels: np.ndarray,
    labs_probs: np.ndarray,
    labs_labels: np.ndarray,
    fusion_probs: np.ndarray,
    fusion_labels: np.ndarray,
    output_dir: str | None = None,
) -> Dict[str, object]:
    """Run evaluations across all modules and fusion.

    Args:
        nlp_probs: NLP predicted probabilities.
        nlp_labels: NLP true labels.
        imaging_probs: Imaging predicted probabilities.
        imaging_labels: Imaging true labels.
        labs_probs: Labs predicted probabilities.
        labs_labels: Labs true labels.
        fusion_probs: Fusion predicted probabilities.
        fusion_labels: Fusion true labels.
        output_dir: Directory to save evaluation JSON. Defaults to `logs/eval/`.

    Returns:
        Dictionary with evaluation results for each module and fusion.
    """
    if output_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
        output_dir = base_dir / "logs" / "eval"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, object] = {}

    # Evaluate each module
    modules = [
        ("nlp", nlp_probs, nlp_labels),
        ("imaging", imaging_probs, imaging_labels),
        ("labs", labs_probs, labs_labels),
        ("fusion", fusion_probs, fusion_labels),
    ]

    for name, probs, labels in modules:
        results[name] = evaluate_module(
            y_true=labels,
            y_probs=probs,
            module_name=name,
            n_bins=10,
            apply_temperature=True,
            apply_isotonic=True,
        )

    # Save results
    out_path = output_dir / "evaluation_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to {out_path}")

    return results


if __name__ == "__main__":
    # Demo with random data
    np.random.seed(42)
    N = 200
    C_nlp = 4
    C_imaging = 2
    C_labs = 3
    C_fusion = 2

    nlp_probs = np.random.rand(N, C_nlp)
    nlp_labels = (np.random.rand(N, C_nlp) > 0.5).astype(int)

    imaging_probs = np.random.rand(N, C_imaging)
    imaging_labels = (np.random.rand(N, C_imaging) > 0.5).astype(int)

    labs_probs = np.random.rand(N, C_labs)
    labs_labels = (np.random.rand(N, C_labs) > 0.5).astype(int)

    fusion_probs = np.random.rand(N, C_fusion)
    fusion_labels = (np.random.rand(N, C_fusion) > 0.5).astype(int)

    evaluate_all_modules(
        nlp_probs=nlp_probs,
        nlp_labels=nlp_labels,
        imaging_probs=imaging_probs,
        imaging_labels=imaging_labels,
        labs_probs=labs_probs,
        labs_labels=labs_labels,
        fusion_probs=fusion_probs,
        fusion_labels=fusion_labels,
    )
