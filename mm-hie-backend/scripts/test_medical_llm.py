"""
Test script for the fine-tuned medical LLM.
Run this to verify the model works correctly.
"""

import sys
sys.path.append('.')

from app.rag.medical_llm import get_medical_llm

def test_medical_llm():
    """Test the fine-tuned medical model."""
    
    print("=" * 80)
    print("Testing Fine-Tuned Medical LLM")
    print("=" * 80)
    
    llm = get_medical_llm()
    
    # Test queries
    test_cases = [
        {
            "type": "Disease to Medicine",
            "query": "What medicine can be used for fever?"
        },
        {
            "type": "Medicine Information",
            "query": "Tell me about Paracetamol"
        },
        {
            "type": "Contraindications",
            "query": "What are the contraindications for Ibuprofen?"
        },
        {
            "type": "Bacterial Infection",
            "query": "Suggest medicine for bacterial infection"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}: {test['type']}")
        print(f"{'=' * 80}")
        print(f"Q: {test['query']}")
        print(f"\nA: ", end="", flush=True)
        
        try:
            response = llm.generate(test['query'], max_new_tokens=200)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 80}")
    print("Testing complete!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    test_medical_llm()
