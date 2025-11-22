import pytest
from unittest.mock import MagicMock, patch
from app.agent.chat_agent_service import ChatAgentService
from app.schemas import CaseCreate, SymptomInput

@pytest.fixture
def mock_orchestrator():
    return MagicMock()

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def service(mock_orchestrator):
    return ChatAgentService(mock_orchestrator)

def test_handle_message_intake(service, mock_db):
    # Mock intent and slots
    with patch("app.agent.chat_agent_service.IntentClassifier") as MockIntent, \
         patch("app.agent.chat_agent_service.SlotExtractor") as MockSlots, \
         patch("app.agent.chat_agent_service.generate_followups") as mock_gen:
        
        MockIntent.return_value.classify.return_value = MagicMock(label="symptom_description")
        MockSlots.return_value.extract.return_value = MagicMock(
            slots={"symptoms": "headache"}, 
            missing=["duration", "severity"]
        )
        
        # Mock generator to return specific questions
        mock_gen.return_value = ["How long?", "How bad?"]
        
        service._orchestrator.create_case.return_value = "case-123"
        
        response = service.handle_message("I have a headache", mock_db)
        
        assert response.case_id == "case-123"
        assert response.action.action == "ask"
        # Check for the mocked question text
        assert "How long?" in response.reply

def test_handle_message_diagnosis(service, mock_db):
    # Mock intent and slots - complete info
    with patch("app.agent.chat_agent_service.IntentClassifier") as MockIntent, \
         patch("app.agent.chat_agent_service.SlotExtractor") as MockSlots, \
         patch("app.agent.chat_agent_service.rag") as mock_rag, \
         patch("app.agent.chat_agent_service.DrugChecker") as MockDrugChecker, \
         patch("app.agent.chat_agent_service.generate_followups") as mock_gen:
        
        MockIntent.return_value.classify.return_value = MagicMock(label="symptom_description")
        MockSlots.return_value.extract.return_value = MagicMock(
            slots={"symptoms": "headache", "duration": "2 days", "severity": "high"}, 
            missing=[]
        )
        
        # Mock generator to return NO followups
        mock_gen.return_value = []
        
        mock_rag.diagnose.return_value = {"answer": "Migraine", "context": []}
        mock_rag.explain.return_value = {"answer": "Vasodilation", "context": []}
        mock_rag.treatment.return_value = {"answer": "Rest", "context": []}
        
        # Mock Drug Checker
        mock_checker_instance = MockDrugChecker.return_value
        drug_mock = MagicMock()
        drug_mock.name = "Paracetamol"
        drug_mock.safe = True
        drug_mock.rationale = "Safe"
        mock_checker_instance.suggest.return_value = [drug_mock]
        service._drug_checker = mock_checker_instance

        response = service.handle_message("I have a severe headache for 2 days", mock_db, case_id="case-123")
        
        assert response.action.action == "info"
        assert response.diagnosis is not None
        assert len(response.diagnosis["drug_suggestions"]) == 1
        assert response.diagnosis["drug_suggestions"][0]["name"] == "Paracetamol"
