from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .calibration import calibration_curve


def plot_reliability_diagram(
    prob_true: List[float],
    prob_pred: List[float],
    n_bins: int,
    title: str = "Reliability Diagram",
    output_path: str | None = None,
) -> None:
    """Plot a reliability diagram.

    Args:
        prob_true: True probability in each bin.
        prob_pred: Mean predicted probability in each bin.
        n_bins: Number of bins.
        title: Plot title.
        output_path: Path to save PNG. If None, show plot.
    """
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved reliability diagram to {output_path}")
    else:
        plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    output_path: str | None = None,
) -> None:
    """Plot calibration curve (fraction of positives vs. mean predicted value).

    Args:
        y_true: True binary labels (N,).
        y_probs: Predicted probabilities (N,).
        n_bins: Number of bins.
        title: Plot title.
        output_path: Path to save PNG. If None, show plot.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
    plot_reliability_diagram(
        prob_true=prob_true,
        prob_pred=prob_pred,
        n_bins=n_bins,
        title=title,
        output_path=output_path,
    )


def plot_all_module_calibrations(
    eval_results: Dict[str, object],
    output_dir: str | None = None,
) -> None:
    """Generate reliability diagrams for all modules from evaluation results.

    Args:
        eval_results: Output from evaluate_all_modules().
        output_dir: Directory to save PNG plots. Defaults to `logs/eval/`.
    """
    if output_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
        output_dir = base_dir / "logs" / "eval"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for module_name, result in eval_results.items():
        if isinstance(result, dict) and "calibration_curve" in result:
            curve = result["calibration_curve"]
            prob_true = curve["prob_true"]
            prob_pred = curve["prob_pred"]
            n_bins = len(prob_true)

            title = f"{module_name.title()} Reliability Diagram"
            png_path = output_dir / f"{module_name}_reliability.png"

            plot_reliability_diagram(
                prob_true=prob_true,
                prob_pred=prob_pred,
                n_bins=n_bins,
                title=title,
                output_path=str(png_path),
            )

    print(f"Calibration plots saved to {output_dir}")


if __name__ == "__main__":
    # Demo: load evaluation results and plot
    base_dir = Path(__file__).resolve().parents[2]
    eval_path = base_dir / "logs" / "eval" / "evaluation_results.json"

    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        plot_all_module_calibrations(results)
    else:
        print(f"Evaluation results not found at {eval_path}. Run evaluate_all.py first.")
