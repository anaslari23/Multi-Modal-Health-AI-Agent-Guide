from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


class PDFReportGenerator:
    def __init__(self, base_dir: str = "./models/reports") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, case_id: str, case_data: Dict[str, Any]) -> Path:
        out_path = self.base_dir / f"case_{case_id}.pdf"
        c = canvas.Canvas(str(out_path), pagesize=A4)
        width, height = A4

        y = height - 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "MM-HIE Clinical Summary")
        y -= 40

        c.setFont("Helvetica", 10)
        meta = case_data.get("meta", {})
        c.drawString(50, y, f"Case ID: {case_id}")
        y -= 15
        c.drawString(50, y, f"Patient ID: {meta.get('patient_id', 'N/A')}")
        y -= 15
        notes = meta.get("notes") or "-"
        c.drawString(50, y, f"Notes: {notes[:80]}")
        y -= 30

        if case_data.get("labs") is not None:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Lab Results")
            y -= 20
            c.setFont("Helvetica", 10)
            for name, lv in case_data["labs"].values.items():
                c.drawString(60, y, f"{name}: {lv.value} ({lv.flag})")
                y -= 15
                if y < 80:
                    c.showPage()
                    y = height - 50

        if case_data.get("imaging") is not None:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Imaging Findings")
            y -= 20
            c.setFont("Helvetica", 10)
            for cls, p in case_data["imaging"].probabilities.items():
                c.drawString(60, y, f"{cls}: {p:.2f}")
                y -= 15
            if case_data["imaging"].gradcam_path:
                c.drawString(60, y, f"Grad-CAM: {case_data['imaging'].gradcam_path}")
                y -= 15

        if case_data.get("nlp") is not None:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "NLP Symptom Insights")
            y -= 20
            c.setFont("Helvetica", 10)
            for cond in case_data["nlp"].conditions:
                c.drawString(60, y, f"{cond.condition}: {cond.prob:.2f}")
                y -= 15

        if case_data.get("vitals") is not None:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Vitals Summary")
            y -= 20
            c.setFont("Helvetica", 10)
            v = case_data["vitals"]
            c.drawString(60, y, f"Vitals risk: {v.vitals_risk:.2f}")
            y -= 15
            c.drawString(60, y, f"Anomalies: {', '.join(v.anomalies) or '-'}")
            y -= 15

        if case_data.get("fusion") is not None:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Final Risk & Triage")
            y -= 20
            c.setFont("Helvetica", 10)
            f = case_data["fusion"]
            c.drawString(60, y, f"Risk score: {f.final_risk_score:.2f}")
            y -= 15
            c.drawString(60, y, f"Triage: {f.triage}")
            y -= 15

        agent_summary = case_data.get("agent_summary")
        if agent_summary:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Agent Summary")
            y -= 20
            c.setFont("Helvetica", 10)
            # Only include a short summary, not the full reasoning chain.
            c.drawString(60, y, agent_summary[:120])
            y -= 15

        if case_data.get("xai") is not None:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Explanation")
            y -= 20
            c.setFont("Helvetica", 10)
            c.drawString(60, y, case_data["xai"].summary[:100])
            y -= 15

        c.showPage()
        c.save()
        return out_path
