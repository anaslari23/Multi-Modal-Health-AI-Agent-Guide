"""LLM integration package for medical reasoning.

Provides a singleton `medical_reasoner` wrapping a large medical reasoning
language model (e.g. dousery/medical-reasoning-gpt-oss-20b).

Usage:
    from app.llm.reasoning_model import medical_reasoner
    answer = medical_reasoner.generate("...prompt...")
"""

from .reasoning_model import MedicalReasoningLLM, medical_reasoner  # noqa: F401
