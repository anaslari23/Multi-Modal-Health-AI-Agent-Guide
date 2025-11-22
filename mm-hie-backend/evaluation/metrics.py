from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn import metrics


def imaging_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Compute AUC, sensitivity, and specificity for imaging predictions.

    Assumes binary classification for simplicity: y_true in {0, 1}, y_scores
    are probabilities for the positive class.
    """

    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    auc = float(metrics.auc(fpr, tpr))

    # Choose threshold at Youden's J statistic for reporting sens/spec.
    j_scores = tpr - fpr
    j_best_idx = int(np.argmax(j_scores))
    best_thresh = thresholds[j_best_idx]

    y_pred = (y_scores >= best_thresh).astype(int)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": float(best_thresh),
    }


def nlp_f1_per_label(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Compute F1-score per label for multi-label NLP outputs.

    y_true and y_pred are binary indicator matrices of shape [N, L].
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    scores: Dict[str, float] = {}
    for idx, label in enumerate(labels):
        f1 = metrics.f1_score(y_true[:, idx], y_pred[:, idx])
        scores[label] = float(f1)
    return scores


def labs_parsing_f1(y_true: List[str], y_pred: List[str]) -> float:
    """Compute F1 for lab parsing treated as token-level labels.

    For a prototype, we treat each lab name/flag pair as a label and compute
    micro-F1 over the set of predicted vs true labels.
    """

    # Convert to sets of labels per example and flatten.
    true_labels: List[str] = []
    pred_labels: List[str] = []

    for t, p in zip(y_true, y_pred):
        # Inputs may already be label strings; split on commas for robustness.
        true_labels.extend([s.strip() for s in str(t).split(",") if s.strip()])
        pred_labels.extend([s.strip() for s in str(p).split(",") if s.strip()])

    all_labels = sorted(set(true_labels) | set(pred_labels))
    if not all_labels:
        return 0.0

    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}

    y_true_bin = np.zeros((len(true_labels), len(all_labels)), dtype=int)
    y_pred_bin = np.zeros((len(pred_labels), len(all_labels)), dtype=int)

    for i, lab in enumerate(true_labels):
        y_true_bin[i, label_to_idx[lab]] = 1
    for i, lab in enumerate(pred_labels):
        y_pred_bin[i, label_to_idx[lab]] = 1

    f1_micro = metrics.f1_score(y_true_bin.flatten(), y_pred_bin.flatten())
    return float(f1_micro)


def fusion_brier_and_calibration(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, List[float]]]:
    """Compute Brier score and calibration curve for fused probabilities.

    Returns
    -------
    brier : float
        Mean squared error between probs and labels.
    calib : dict
        Dictionary with fields `bin_centers`, `prob_mean`, `freq_mean` for
        plotting a calibration curve.
    """

    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    brier = metrics.brier_score_loss(y_true, y_probs)

    prob_true, prob_pred = metrics.calibration_curve(y_true, y_probs, n_bins=n_bins)

    calib = {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }
    return float(brier), calib


def risk_auroc(y_true: np.ndarray, risk_scores: np.ndarray) -> float:
    """Compute AUROC for scalar risk scores (e.g., final risk / triage).

    y_true: binary event indicator.
    risk_scores: continuous risk scores.
    """

    y_true = np.asarray(y_true).astype(int)
    risk_scores = np.asarray(risk_scores).astype(float)

    fpr, tpr, _ = metrics.roc_curve(y_true, risk_scores)
    auc = metrics.auc(fpr, tpr)
    return float(auc)
