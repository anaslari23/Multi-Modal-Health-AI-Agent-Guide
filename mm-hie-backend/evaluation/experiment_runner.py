from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from .metrics import (
    imaging_metrics,
    nlp_f1_per_label,
    labs_parsing_f1,
    fusion_brier_and_calibration,
    risk_auroc,
)


class ExperimentRunner:
    """Offline evaluation runner for all modalities.

    This is a prototype that expects precomputed predictions and labels stored
    in simple CSV/JSON files under a data directory. It computes metrics for
    imaging, NLP, labs, fusion, and risk prediction, writes a JSON summary, and
    emits calibration/ROC plots.
    """

    def __init__(self, data_dir: str = "./evaluation/data", out_dir: str = "./evaluation/results") -> None:
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        metrics_out: Dict[str, Any] = {}

        # Imaging
        imaging_file = self.data_dir / "imaging_eval.npz"
        if imaging_file.exists():
            arr = np.load(imaging_file)
            y_true = arr["y_true"]
            y_scores = arr["y_scores"]
            metrics_out["imaging"] = imaging_metrics(y_true, y_scores)

        # NLP
        nlp_file = self.data_dir / "nlp_eval.npz"
        if nlp_file.exists():
            arr = np.load(nlp_file, allow_pickle=True)
            y_true = arr["y_true"]
            y_pred = arr["y_pred"]
            labels = list(arr["labels"])
            metrics_out["nlp_f1"] = nlp_f1_per_label(y_true, y_pred, labels)

        # Labs parsing
        labs_file = self.data_dir / "labs_eval.json"
        if labs_file.exists():
            with labs_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            y_true = data.get("true", [])
            y_pred = data.get("pred", [])
            metrics_out["labs_f1"] = labs_parsing_f1(y_true, y_pred)

        # Fusion Brier + calibration and risk AUROC
        fusion_file = self.data_dir / "fusion_eval.npz"
        if fusion_file.exists():
            arr = np.load(fusion_file)
            y_true = arr["y_true"]
            y_probs = arr["y_probs"]
            brier, calib = fusion_brier_and_calibration(y_true, y_probs)
            metrics_out["fusion_brier"] = brier
            metrics_out["fusion_calibration"] = calib

            self._plot_calibration(calib, self.out_dir / "fusion_calibration.png")

        risk_file = self.data_dir / "risk_eval.npz"
        if risk_file.exists():
            arr = np.load(risk_file)
            y_true = arr["y_true"]
            risk_scores = arr["risk_scores"]
            metrics_out["risk_auroc"] = risk_auroc(y_true, risk_scores)

        # Save metrics JSON
        out_json = self.out_dir / "metrics_summary.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)

        return metrics_out

    def _plot_calibration(self, calib: Dict[str, Any], out_path: Path) -> None:
        prob_true = np.asarray(calib["prob_true"], dtype=float)
        prob_pred = np.asarray(calib["prob_pred"], dtype=float)

        plt.figure(figsize=(4, 4))
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title("Calibration curve")
        plt.legend()
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()


if __name__ == "__main__":
    runner = ExperimentRunner()
    results = runner.run()
    print(json.dumps(results, indent=2))
