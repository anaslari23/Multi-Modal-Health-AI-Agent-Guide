from __future__ import annotations

import uuid
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any

from sqlalchemy import String, Integer, DateTime, ForeignKey, Float, Enum
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    patient_name: Mapped[str] = mapped_column(String, nullable=False)
    patient_age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    patient_gender: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("in_progress", "completed", name="case_status"),
        default="in_progress",
        nullable=False,
    )

    modalities: Mapped[List["Modality"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    xai_outputs: Mapped[List["XAIOutput"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    agent_actions: Mapped[List["AgentAction"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    reports: Mapped[List["Report"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )


class Modality(Base):
    __tablename__ = "modalities"

    id: Mapped[uuid.UUID] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        String, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(
        Enum("symptoms", "labs", "imaging", "vitals", name="modality_type"),
        nullable=False,
    )
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    processed: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    case: Mapped[Case] = relationship(back_populates="modalities")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        String, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False
    )
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    triage: Mapped[str] = mapped_column(String, nullable=False)
    conditions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    fusion_vector: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    agent_summary: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    case: Mapped[Case] = relationship(back_populates="analysis_results")


class XAIOutput(Base):
    __tablename__ = "xai_outputs"

    id: Mapped[uuid.UUID] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        String, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False
    )
    labs_shap_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    nlp_highlights: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    gradcam_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    explanation_text: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    case: Mapped[Case] = relationship(back_populates="xai_outputs")


class AgentAction(Base):
    __tablename__ = "agent_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        String, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False
    )
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    action_type: Mapped[str] = mapped_column(String, nullable=False)
    action_detail: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    confidence_gain: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    case: Mapped[Case] = relationship(back_populates="agent_actions")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        String, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False
    )
    pdf_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    case: Mapped[Case] = relationship(back_populates="reports")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        String, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False
    )
    event: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    case: Mapped[Case] = relationship(back_populates="audit_logs")
