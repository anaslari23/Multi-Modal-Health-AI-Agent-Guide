"""
Test script for medical reasoner (Ollama).
"""
import sys
sys.path.append('.')

from app.llm.reasoning_model import medical_reasoner

import logging
logging.basicConfig(level=logging.INFO)

def test_reasoner():
    print("Testing Medical Reasoner (Ollama)...")
    try:
        response = medical_reasoner.generate("Hello, are you a doctor?", max_tokens=50)
        print(f"Response: {response}")
        print("✅ Reasoner is working!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_reasoner()
