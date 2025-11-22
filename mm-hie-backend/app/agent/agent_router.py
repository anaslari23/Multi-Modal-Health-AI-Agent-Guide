from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..orchestrator_instance import orchestrator
from .action_schema import ChatAgentResponse
from .chat_agent_service import ChatAgentService


router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    case_id: Optional[str] = None
    message: str


@router.post("/chat", response_model=ChatAgentResponse)
def agent_chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatAgentResponse:
    service = ChatAgentService(orchestrator)
    return service.handle_message(message=payload.message, db=db, case_id=payload.case_id)
