from __future__ import annotations

from typing import Optional, List

from sqlalchemy.orm import Session

from ..orchestrator import Orchestrator
from ..rag.rag_engine import rag
from ..schemas import CaseCreate, SymptomInput
from .action_schema import AgentAction, ChatAgentResponse
from .drug_checker import DrugChecker
from .slot_extractor import SlotExtractor
from ..llm.reasoning_model import medical_reasoner


DOCTOR_SYSTEM_PROMPT = """You are an expert clinical AI assistant.
Your goal is to diagnose the patient based on their symptoms.
- If the patient's description is vague or incomplete (missing onset, duration, severity, etc.), ask specific clarifying questions ONE BY ONE.
- Do NOT ask multiple questions at once.
- If you have sufficient information, provide a differential diagnosis, potential causes, and recommended next steps.
- Be empathetic, professional, and concise.
- If suggesting medication, mention generic names and safety warnings.
"""


class ChatAgentService:
    """High-level conversational agent for the MM-HIE frontend.

    Uses a Large Language Model (LLM) to drive the clinical interview,
    replacing rigid rule-based slot extraction with natural conversation.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator
        self._slots = SlotExtractor()  # Keep for background structured data extraction
        self._drug_checker = DrugChecker()

    def handle_message(
        self,
        message: str,
        db: Session,
        case_id: Optional[str] = None,
    ) -> ChatAgentResponse:
        text = message.strip()

        # 1. Ensure case exists
        if case_id is None:
            # Create ad-hoc case
            case = CaseCreate(patient_id="chat-user", notes=text)
            case_id = self._orchestrator.create_case(case, db)

        # 2. Extract slots in background (for DB/structured analysis)
        slot_result = self._slots.extract(text)
        if slot_result.slots.get("symptoms"):
            payload = SymptomInput(text=slot_result.slots["symptoms"], top_n=5)
            self._orchestrator.add_symptoms(case_id, payload, db)

        # 3. Build Context for LLM
        # Ideally we would fetch full chat history. For now, we use the current message
        # and any accumulated structured data from the case.
        case_data = self._orchestrator._get_case(case_id)
        
        # Construct the prompt
        # Format: System + Context + User Input
        symptoms_known = case_data.get("nlp")
        symptoms_text = ""
        if symptoms_known and symptoms_known.conditions:
             top_conditions = ", ".join([c.condition for c in symptoms_known.conditions[:3]])
             symptoms_text = f"Current capabilities detect possible: {top_conditions}."

        prompt = f"""{DOCTOR_SYSTEM_PROMPT}

Context:
Patient ID: {case_id}
{symptoms_text}

Patient: "{text}"
Doctor:"""

        # 4. Generate Response via LLM
        try:
            # "Train" / prompt the model to be a doctor
            response_text = medical_reasoner.generate(
                prompt,
                max_tokens=256,
                temperature=0.7
            )
            reply = response_text.strip()
        except Exception as e:
            # Fallback if LLM fails (e.g. model not loaded/OOM)
            reply = f"I'm having trouble accessing my medical reasoning module ({e}). Please describe your symptoms in more detail."

        # 5. Determine Action Type based on response
        # If the LLM asks a question (ends with ?), it's an 'ask' action.
        # If it gives a diagnosis/plan, it's an 'info' action.
        if "?" in reply:
            action_type = "ask"
            action_text = "Asking clarifying question."
        else:
            action_type = "info"
            action_text = "Providing clinical assessment."
        
        # 6. Construct Response
        action = AgentAction(
            action=action_type,
            text=action_text,
            followups=[], 
            metadata={"slots": slot_result.slots},
        )

        return ChatAgentResponse(reply=reply, action=action, case_id=case_id)
