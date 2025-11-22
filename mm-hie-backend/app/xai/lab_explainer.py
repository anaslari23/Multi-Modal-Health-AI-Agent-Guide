from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

# Use a non-interactive backend suitable for server/test environments.
matplotlib.use("Agg")  # type: ignore
import matplotlib.pyplot as plt

from ..schemas import LabResults
from ..utils.s3_client import get_s3_client


class LabExplainer:
    """SHAP-style explainer for lab values.

    This is a lightweight, deterministic approximation that:
    - Derives per-lab contribution scores from flags (HIGH/LOW/NORMAL).
    - Builds a simple waterfall-style bar plot as a PNG.
    - Returns a short text summary of the most influential labs.
    """

    def __init__(self, base_dir: str = "./models/xai_labs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _compute_contributions(self, labs: LabResults) -> Dict[str, float]:
        contribs: Dict[str, float] = {}
        for name, lv in labs.values.items():
            flag = (lv.flag or "").upper()
            if flag == "HIGH":
                contribs[name] = 1.0
            elif flag == "LOW":
                contribs[name] = -1.0
            else:
                contribs[name] = 0.0
        return contribs

    def _build_summary(self, contribs: Dict[str, float]) -> str:
        if not contribs:
            return "No lab explainability available."

        # Rank by absolute contribution.
        ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top = [item for item in ranked if abs(item[1]) > 0.0][:5]
        if not top:
            return "All lab values were within normal range; minimal impact on risk."

        parts = []
        for name, score in top:
            direction = "increased" if score > 0 else "decreased"
            parts.append(f"{name} {direction} risk")
        return "; ".join(parts) + "."

    def _save_waterfall(self, case_id: str, contribs: Dict[str, float]) -> Path:
        if not contribs:
            # Still create an empty placeholder plot for consistency.
            plt.figure(figsize=(4, 2))
            plt.title("Lab explainability")
            out_path = self.base_dir / f"labs_shap_{case_id}.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            return out_path

        ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        names = [k for k, _ in ranked]
        scores = [v for _, v in ranked]

        fig, ax = plt.subplots(figsize=(6, max(2, len(names) * 0.3)))
        colors = ["tab:red" if s > 0 else "tab:blue" for s in scores]
        ax.barh(names, scores, color=colors)
        ax.set_xlabel("Contribution (arbitrary units)")
        ax.set_title("Lab SHAP-style contributions")
        ax.axvline(0.0, color="black", linewidth=0.8)
        fig.tight_layout()

        out_path = self.base_dir / f"labs_shap_{case_id}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def explain(self, case_id: str, labs: LabResults) -> Tuple[str, str]:
        """Compute lab contributions, save a waterfall plot, and summarise.

        Returns (summary_text, png_path_str).
        """

        contribs = self._compute_contributions(labs)
        summary = self._build_summary(contribs)
        png_path = self._save_waterfall(case_id, contribs)

        # Upload to S3 if configured; fall back to local path if S3 is unavailable.
        try:
            s3 = get_s3_client()
            key = f"xai/labs_shap_{case_id}.png"
            s3_key = s3.upload_file(png_path, key)
            return summary, s3_key
        except Exception:
            return summary, str(png_path)
