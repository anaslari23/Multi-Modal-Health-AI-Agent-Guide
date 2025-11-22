from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics


def binary_classification_metrics(y_true, y_scores) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    auc = metrics.auc(fpr, tpr)

    j = tpr - fpr
    idx = int(np.argmax(j))
    thr = thresholds[idx]
    y_pred = (y_scores >= thr).astype(int)
    acc = metrics.accuracy_score(y_true, y_pred)

    return {"auc": float(auc), "accuracy": float(acc), "threshold": float(thr)}
