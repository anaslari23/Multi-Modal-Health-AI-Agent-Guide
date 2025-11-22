from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """Structured action for the frontend to interpret.

    This is intentionally generic but expressive enough to cover:
    - follow-up questions
    - upload requests (labs, imaging, vitals)
    - informational messages
    - medication recommendations
    - escalation / safety banners
    """

    action: str = Field(..., description="High-level action type, e.g. ask, request_upload, info, recommend_meds, escalate")
    text: Optional[str] = Field(
        None,
        description="Optional natural language description of what the agent is doing.",
    )
    followups: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions shown as quick-reply buttons.",
    )
    upload_type: Optional[str] = Field(
        None,
        description="If action=request_upload, this specifies image|lab_pdf|vitals.",
    )
    buttons: List[str] = Field(
        default_factory=list,
        description="Generic buttons/options that the UI can surface to the user.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured payload with diagnosis, tests, meds, etc.",
    )


class ChatMessage(BaseModel):
    """Server-side representation of a chat message in a case timeline."""

    id: str
    case_id: Optional[str] = None
    sender: str  # "user" | "doctor" | "system"
    text: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatAgentResponse(BaseModel):
    """Top-level response contract for /agent/chat.

    This is designed to match the Flutter UI expectations.
    """

    reply: str
    action: AgentAction
    case_id: str
    diagnosis: Optional[Dict[str, Any]] = None

