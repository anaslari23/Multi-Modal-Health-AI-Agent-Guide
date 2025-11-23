import sys
import os
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.agent.chat_agent_service import ChatAgentService
from app.orchestrator import Orchestrator
from app.database import SessionLocal
from app.schemas import CaseCreate

def test_mag():
    print("Initializing ChatAgentService...")
    # Mock orchestrator
    orchestrator = Orchestrator()
    service = ChatAgentService(orchestrator)
    
    db = SessionLocal()
    
    # Create a dummy case
    print("Creating dummy case...")
    case = CaseCreate(patient_id="test-user", notes="Test case")
    case_id = orchestrator.create_case(case, db)
    
    message = "I have a high fever and headache."
    print(f"\nUser Message: {message}")
    
    print("\nProcessing message...")
    response = service.handle_message(message, db, case_id=case_id)
    
    print("\nResponse:")
    print(response.reply)
    
    print("\nAction:")
    print(response.action)

if __name__ == "__main__":
    test_mag()
