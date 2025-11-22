import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "mm-hie-backend"))

from app.orchestrator import Orchestrator
from app.schemas import CaseCreate
from app.database import SessionLocal, engine, Base

# Mock DB session
Base.metadata.create_all(bind=engine)
db = SessionLocal()

def reproduce():
    orch = Orchestrator()
    
    # Create a case
    case_in = CaseCreate(patient_id="test_patient", notes="Test notes")
    case_id = orch.create_case(case_in, db)
    print(f"Created case: {case_id}")
    
    # Run analysis immediately (no modalities)
    print("Running analysis...")
    try:
        orch.run_analysis(case_id, db)
        print("Analysis successful")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
