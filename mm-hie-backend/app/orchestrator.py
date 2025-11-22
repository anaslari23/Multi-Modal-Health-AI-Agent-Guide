from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Any

import logging
from fastapi import UploadFile
from sqlalchemy.orm import Session

from .schemas import (
    CaseCreate,
    SymptomInput,
    VitalsInput,
    AnalysisResponse,
    ReportResponse,
    ConditionProb,
    AgentPosteriorTimelineResponse,
    PosteriorHistoryEntry,
    AgentActionState,
)
from .modules.nlp.model_clinical_bert import ClinicalBERTSymptomModel
from .modules.imaging.imaging_model import ImagingModel
from .modules.timeseries.vitals_transformer import VitalsTransformerModel
from .fusion.multimodal_transformer import MultimodalTransformerFusion
from .fusion.bayes_updater import BayesianUpdater
from .xai.xai_router import XAIAggregator
from .utils.pdf_report import PDFReportGenerator
from .utils.s3_client import get_s3_client
from .agent_brain import AgentBrain
from .llm.reasoning_model import medical_reasoner  # TODO: use for advanced agent reasoning or summaries
from . import crud


logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self) -> None:
        self.cases: Dict[str, Dict[str, Any]] = {}
        self.base_path = Path("./models")

        # Heavy models are **not** constructed at startup. They are
        # lazily instantiated on first use so that FastAPI/Uvicorn reload
        # remains fast and local development on macOS ARM is stable.
        self.symptom_model: Optional[ClinicalBERTSymptomModel] = None
        self.lab_parser = None
        self.imaging_model: Optional[ImagingModel] = None
        self.vitals_model: Optional[VitalsTransformerModel] = None
        self.fusion_model: Optional[MultimodalTransformerFusion] = None
        self.bayes = BayesianUpdater()
        self.xai = XAIAggregator()
        self.report_gen = PDFReportGenerator()
        self.agent = AgentBrain()

    # --- Lazy model initialisers -------------------------------------------------

    def _ensure_symptom_model(self) -> None:
        if self.symptom_model is None:
            logger.info("Initialising ClinicalBERTSymptomModel lazily")
            self.symptom_model = ClinicalBERTSymptomModel()

    def _ensure_imaging_model(self) -> None:
        if self.imaging_model is None:
            logger.info("Initialising ImagingModel lazily")
            self.imaging_model = ImagingModel()

    def _ensure_vitals_model(self) -> None:
        if self.vitals_model is None:
            logger.info("Initialising VitalsTransformerModel lazily")
            self.vitals_model = VitalsTransformerModel()

    def _ensure_fusion_model(self) -> None:
        if self.fusion_model is None:
            logger.info("Initialising MultimodalTransformerFusion lazily")
            self.fusion_model = MultimodalTransformerFusion()

    def create_case(self, case: CaseCreate, db: Session) -> str:
        logger.info(f"Creating case for patient_id={case.patient_id}")
        db_case = crud.create_case(
            db,
            patient_name=case.patient_id,
            patient_age=None,
            patient_gender=None,
        )
        case_id = str(db_case.id)
        logger.info(f"Created case with id={case_id}")
        self.cases[case_id] = {
            "meta": case.dict(),
            "nlp": None,
            "labs": None,
            "imaging": None,
            "vitals": None,
            "fusion": None,
            "xai": None,
            "posterior": None,
            "posterior_history": [],
            "agent_actions": [],
        }

        crud.log_event(db, db_case.id, event="case_created", description="Case created via API")
        return case_id

    def _get_case(self, case_id: str) -> Dict[str, Any]:
        """Get case from in-memory cache, initializing if not present."""
        if case_id not in self.cases:
            # Case exists in DB but not in memory (e.g., after server restart)
            # Initialize empty case structure
            self.cases[case_id] = {
                "meta": {},
                "nlp": None,
                "labs": None,
                "imaging": None,
                "vitals": None,
                "fusion": None,
                "xai": None,
                "posterior": None,
                "posterior_history": [],
                "agent_actions": [],
            }
        return self.cases[case_id]

    def _recompute_posterior(self, case: Dict[str, Any]) -> None:
        """Recompute per-disease posterior using available modality outputs.

        - Prior: symptom NLP condition probabilities.
        - Likelihoods: currently imaging probabilities (others treated as neutral).
        """

        nlp_out = case.get("nlp")
        if nlp_out is None:
            return

        prior = {c.condition: c.prob for c in nlp_out.conditions}

        imaging = case.get("imaging")
        imaging_like = imaging.probabilities if imaging is not None else {}

        posterior_dict = self.bayes.update(prior, imaging_like)
        if not posterior_dict:
            case["posterior"] = None
            return

        posterior = [
            ConditionProb(condition=k, prob=float(v)) for k, v in posterior_dict.items()
        ]
        posterior = sorted(posterior, key=lambda c: c.prob, reverse=True)

        case["posterior"] = posterior
        history = case.setdefault("posterior_history", [])
        history.append(
            {
                "step": len(history) + 1,
                "modalities": {
                    "nlp": case.get("nlp") is not None,
                    "labs": case.get("labs") is not None,
                    "imaging": case.get("imaging") is not None,
                    "vitals": case.get("vitals") is not None,
                },
                "posterior": posterior,
            }
        )

    def get_agent_timeline(self, case_id: str) -> AgentPosteriorTimelineResponse:
        case = self._get_case(case_id)

        raw_history = case.get("posterior_history") or []
        history: list[PosteriorHistoryEntry] = []
        for entry in raw_history:
            raw_posterior = entry.get("posterior") or []
            posterior: list[ConditionProb] = []
            for c in raw_posterior:
                if isinstance(c, ConditionProb):
                    posterior.append(c)
                elif isinstance(c, dict):
                    posterior.append(ConditionProb(**c))
            history.append(
                PosteriorHistoryEntry(
                    step=entry.get("step", 0),
                    modalities=entry.get("modalities") or {},
                    posterior=posterior,
                )
            )

        raw_actions = case.get("agent_actions") or []
        actions: list[AgentActionState] = []
        for a in raw_actions:
            actions.append(
                AgentActionState(
                    trigger=a.get("trigger", ""),
                    next_steps=list(a.get("next_steps") or []),
                    agent_reasoning=a.get("agent_reasoning", ""),
                    confidence_gain=a.get("confidence_gain"),
                )
            )

        return AgentPosteriorTimelineResponse(
            posterior_history=history,
            agent_actions=actions,
        )

    def _run_agent_step(self, case: Dict[str, Any], trigger: str) -> Dict[str, Any]:
        """Invoke the AgentBrain on the current case state and record the action.

        The returned payload has shape:
        {
          "next_steps": [...],
          "agent_reasoning": str,
          "confidence_gain": float,
        }
        """

        agent_out = self.agent.reason(case)

        # TODO: In a future iteration, augment AgentBrain with LLM-based
        # reasoning using `medical_reasoner.generate(...)` or a remote LLM
        # microservice call (see app.llm.reasoning_service). This keeps the
        # current deterministic logic as the primary driver while allowing
        # operators to enable richer natural language reasoning summaries
        # on capable hardware.
        actions = case.setdefault("agent_actions", [])
        actions.append(
            {
                "trigger": trigger,
                "next_steps": agent_out["next_steps"],
                "agent_reasoning": agent_out["agent_reasoning"],
                "confidence_gain": agent_out["confidence_gain"],
            }
        )
        return agent_out

    def _build_agent_summary(self, case: Dict[str, Any]) -> Optional[str]:
        actions = case.get("agent_actions") or []
        if not actions:
            return None

        sorted_actions = sorted(
            actions,
            key=lambda a: (a.get("confidence_gain") or 0.0),
            reverse=True,
        )
        top_actions = sorted_actions[:2]

        parts = []
        for a in top_actions:
            cg = a.get("confidence_gain")
            steps = a.get("next_steps") or []
            if steps:
                headline = steps[0]
            else:
                reasoning = a.get("agent_reasoning") or ""
                headline = reasoning.splitlines()[0] if reasoning else ""

            if cg is not None:
                parts.append(f"{a.get('trigger')}: {headline} (Δconfidence {cg:.2f})")
            else:
                parts.append(f"{a.get('trigger')}: {headline}")

        return " | ".join(parts)

    def add_symptoms(self, case_id: str, payload: SymptomInput, db: Session):
        logger.info(f"Adding symptoms to case_id={case_id}")
        # Get case from memory first (auto-initializes if needed)
        case = self._get_case(case_id)
        logger.info(f"Retrieved case from memory")
        
        # Verify case exists in database
        db_case = crud.get_case(db, case_id)
        if db_case is None:
            raise KeyError(case_id)
        logger.info(f"Verified case exists in DB")

        logger.info(f"Ensuring symptom model is loaded...")
        self._ensure_symptom_model()
        assert self.symptom_model is not None  # for type checkers
        logger.info(f"Running NLP inference on text: {payload.text[:50]}...")
        nlp_out = self.symptom_model.infer(payload.text, top_n=payload.top_n)
        logger.info(f"NLP inference complete")
        case["nlp"] = nlp_out

        # Update posterior after symptoms.
        self._recompute_posterior(case)

        # Run agent reasoning step.
        agent_out = self._run_agent_step(case, trigger="symptoms")

        crud.add_modality(
            db,
            case_id=db_case.id,
            modality_type="symptoms",
            payload={"input": payload.dict()},
            processed=nlp_out.dict(),
        )
        crud.save_agent_action(
            db,
            case_id=db_case.id,
            step_number=1,
            action_type="add_symptoms",
            action_detail={
                "text": payload.text,
                "top_n": payload.top_n,
                "agent": agent_out,
            },
            confidence_gain=agent_out["confidence_gain"],
        )
        crud.log_event(db, db_case.id, event="symptoms_added", description="Symptoms added to case")

        return nlp_out

    async def add_lab_report(self, case_id: str, file: UploadFile, db: Session):
        db_case = crud.get_case(db, case_id)
        if db_case is None:
            raise KeyError(case_id)

        case = self._get_case(case_id)
        content = await file.read()

        if self.lab_parser is None:
            # Lazy import to avoid pulling PaddleOCR (and paddle) into the
            # process at startup. This keeps reloads fast and is friendlier
            # to macOS ARM setups where Paddle may not be present.
            from .modules.ocr_parser.lab_ocr import LabOCRParser

            self.lab_parser = LabOCRParser()

        labs = self.lab_parser.parse_bytes(content)
        case["labs"] = labs

        # Update posterior after labs (currently neutral likelihood, but tracked in history).
        self._recompute_posterior(case)

        # Run agent reasoning step.
        agent_out = self._run_agent_step(case, trigger="labs")

        crud.add_modality(
            db,
            case_id=db_case.id,
            modality_type="labs",
            payload=None,
            processed=labs.dict(),
        )
        crud.save_agent_action(
            db,
            case_id=db_case.id,
            step_number=2,
            action_type="add_labs",
            action_detail={
                "source": "upload-report",
                "agent": agent_out,
            },
            confidence_gain=agent_out["confidence_gain"],
        )
        crud.log_event(db, db_case.id, event="labs_added", description="Lab report processed and added")

        return labs

    async def add_image(self, case_id: str, file: UploadFile, db: Session):
        db_case = crud.get_case(db, case_id)
        if db_case is None:
            raise KeyError(case_id)

        case = self._get_case(case_id)
        content = await file.read()

        # Lazily construct the imaging model so that importing the
        # orchestrator (and thus the FastAPI app) is cheap.
        self._ensure_imaging_model()
        assert self.imaging_model is not None  # for type checkers

        imaging_out = self.imaging_model.infer_bytes(content, case_id=case_id)
        case["imaging"] = imaging_out

        # Update posterior with imaging likelihoods.
        self._recompute_posterior(case)

        # Run agent reasoning step.
        agent_out = self._run_agent_step(case, trigger="imaging")

        crud.add_modality(
            db,
            case_id=db_case.id,
            modality_type="imaging",
            payload=None,
            processed=imaging_out.dict(),
        )
        crud.save_agent_action(
            db,
            case_id=db_case.id,
            step_number=3,
            action_type="add_image",
            action_detail={
                "source": "upload-image",
                "agent": agent_out,
            },
            confidence_gain=agent_out["confidence_gain"],
        )
        crud.log_event(db, db_case.id, event="imaging_added", description="Imaging data processed and added")

        return imaging_out

    def add_vitals(self, case_id: str, payload: VitalsInput, db: Session):
        db_case = crud.get_case(db, case_id)
        if db_case is None:
            raise KeyError(case_id)

        case = self._get_case(case_id)

        self._ensure_vitals_model()
        assert self.vitals_model is not None  # for type checkers

        vitals_out = self.vitals_model.infer(
            heart_rate=payload.heart_rate,
            spo2=payload.spo2,
            temperature=payload.temperature,
            resp_rate=payload.resp_rate,
        )
        case["vitals"] = vitals_out

        # Update posterior after vitals (currently treated as neutral likelihood).
        self._recompute_posterior(case)

        # Run agent reasoning step.
        agent_out = self._run_agent_step(case, trigger="vitals")

        crud.add_modality(
            db,
            case_id=db_case.id,
            modality_type="vitals",
            payload={"input": payload.dict()},
            processed=vitals_out.dict(),
        )
        crud.save_agent_action(
            db,
            case_id=db_case.id,
            step_number=4,
            action_type="add_vitals",
            action_detail={"source": "vitals", "agent": agent_out},
            confidence_gain=agent_out["confidence_gain"],
        )
        crud.log_event(db, db_case.id, event="vitals_added", description="Vitals added to case")

        return vitals_out

    def run_analysis(self, case_id: str, db: Session) -> AnalysisResponse:
        db_case = crud.get_case(db, case_id)
        if db_case is None:
            raise KeyError(case_id)

        case = self._get_case(case_id)

        self._ensure_fusion_model()
        assert self.fusion_model is not None  # for type checkers

        fusion_out = self.fusion_model.fuse(
            nlp=case["nlp"],
            labs=case["labs"],
            imaging=case["imaging"],
            vitals=case["vitals"],
        )

        risk_score = fusion_out["risk_score"]
        if risk_score >= 70:
            triage = "High"
        elif risk_score >= 40:
            triage = "Medium"
        else:
            triage = "Low"

        conditions = case["nlp"].conditions if case["nlp"] is not None else []
        conditions = sorted(conditions, key=lambda c: c.prob, reverse=True)

        from .schemas import FusionOutput

        # Derive simple modality contribution scores in [0, 1] for UI visualisation.
        nlp_score = 1.0 if case["nlp"] is not None else 0.0
        imaging_score = 0.0
        if case["imaging"] is not None and case["imaging"].probabilities:
            imaging_score = float(max(case["imaging"].probabilities.values()))
        lab_score = 0.0
        if case["labs"] is not None and case["labs"].values:
            abnormal = [lv for lv in case["labs"].values.values() if lv.flag.upper() != "NORMAL"]
            lab_score = 1.0 if abnormal else 0.2
        vitals_score = 0.0
        if case["vitals"] is not None:
            vitals_score = float(case["vitals"].vitals_risk)

        modality_scores = {
            "nlp": max(0.0, min(1.0, nlp_score)),
            "imaging": max(0.0, min(1.0, imaging_score)),
            "labs": max(0.0, min(1.0, lab_score)),
            "vitals": max(0.0, min(1.0, vitals_score)),
        }

        fusion = FusionOutput(
            final_risk_score=risk_score,
            triage=triage,
            conditions=conditions,
            modality_scores=modality_scores,
        )

        try:
            xai = self.xai.explain(case_id=case_id, case_data=case, fusion=fusion)
        except Exception as e:
            logger.error(f"XAI generation failed: {e}")
            from .schemas import XAIOutput
            xai = XAIOutput(
                summary=f"Explanation unavailable due to internal error: {e}",
                gradcam_path=None,
                labs_shap_path=None,
                nlp_highlights=None,
            )

        # Derive signed URLs for S3-hosted artefacts when possible.
        try:
            s3 = get_s3_client()
        except Exception:
            s3 = None

        if s3 is not None:
            # Grad-CAM
            if case.get("imaging") is not None and case["imaging"].gradcam_path:
                try:
                    url = s3.generate_presigned_url(case["imaging"].gradcam_path)
                    case["imaging"].gradcam_url = url
                except Exception:
                    pass
            if xai.gradcam_path:
                try:
                    xai.gradcam_url = s3.generate_presigned_url(xai.gradcam_path)
                except Exception:
                    pass

            # Labs SHAP PNG
            if xai.labs_shap_path:
                try:
                    xai.labs_shap_url = s3.generate_presigned_url(xai.labs_shap_path)
                except Exception:
                    pass

        case["fusion"] = fusion
        case["xai"] = xai

        agent_summary = self._build_agent_summary(case)
        case["agent_summary"] = agent_summary

        crud.save_analysis_result(
            db,
            case_id=db_case.id,
            risk_score=risk_score,
            triage=triage,
            conditions={"conditions": [c.dict() for c in conditions]},
            fusion_vector={
                "fused_vector": fusion_out["fused_vector"].tolist(),
                "disease_probs": fusion_out["disease_probs"].tolist(),
                "risk_score": risk_score,
            },
            agent_summary=agent_summary,
        )

        crud.save_xai_output(
            db,
            case_id=db_case.id,
            labs_shap_path=xai.labs_shap_path,
            nlp_highlights=None,
            gradcam_path=xai.gradcam_path,
            explanation_text=xai.summary,
        )

        crud.save_agent_action(
            db,
            case_id=db_case.id,
            step_number=5,
            action_type="run_analysis",
            action_detail={"result": fusion.dict()},
            confidence_gain=fusion.final_risk_score,
        )
        crud.log_event(db, db_case.id, event="analysis_run", description="Full multimodal analysis executed")

        posterior = case.get("posterior")

        return AnalysisResponse(
            nlp=case["nlp"],
            labs=case["labs"],
            imaging=case["imaging"],
            vitals=case["vitals"],
            fusion=fusion,
            xai=xai,
            posterior_probabilities=posterior,
            agent_summary=agent_summary,
        )

    def generate_report(self, case_id: str, db: Session) -> ReportResponse:
        db_case = crud.get_case(db, case_id)
        if db_case is None:
            raise KeyError(case_id)

        case = self._get_case(case_id)
        pdf_path = self.report_gen.generate(case_id=case_id, case_data=case)

        # Upload PDF to S3 if available and derive signed URL.
        pdf_url: Optional[str] = None
        try:
            s3 = get_s3_client()
            key = f"reports/case_{case_id}.pdf"
            s3.upload_file(pdf_path, key)
            pdf_url = s3.generate_presigned_url(key)
        except Exception:
            pdf_url = None

        crud.save_pdf_report(db, case_id=db_case.id, pdf_path=str(pdf_path))
        crud.log_event(db, db_case.id, event="report_generated", description="PDF report generated")
        return ReportResponse(case_id=case_id, pdf_path=str(pdf_path), pdf_url=pdf_url)
