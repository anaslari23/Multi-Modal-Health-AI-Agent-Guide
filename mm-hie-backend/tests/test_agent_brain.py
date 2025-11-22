from __future__ import annotations

from app.agent_brain import AgentBrain
from app.schemas import ConditionProb


def _make_case(with_nlp=True, with_labs=False, with_imaging=False, with_vitals=False, with_posterior=False):
    case = {
        "meta": {"patient_id": "p1"},
        "nlp": None,
        "labs": None,
        "imaging": None,
        "vitals": None,
        "posterior": None,
    }

    if with_nlp:
        case["nlp"] = object()  # we only check presence, not content
    if with_labs:
        case["labs"] = object()
    if with_imaging:
        case["imaging"] = object()
    if with_vitals:
        case["vitals"] = object()
    if with_posterior:
        case["posterior"] = [
            ConditionProb(condition="Pneumonia", prob=0.8),
            ConditionProb(condition="Sepsis", prob=0.2),
        ]

    return case


def test_agent_brain_basic_output_shape():
    agent = AgentBrain()
    case = _make_case()

    out = agent.reason(case)

    assert set(out.keys()) == {"next_steps", "agent_reasoning", "confidence_gain"}
    assert isinstance(out["next_steps"], list)
    assert isinstance(out["agent_reasoning"], str)
    assert isinstance(out["confidence_gain"], float)


def test_agent_brain_suggests_missing_modalities():
    agent = AgentBrain()
    # Only NLP present; others missing
    case = _make_case(with_nlp=True, with_labs=False, with_imaging=False, with_vitals=False)

    out = agent.reason(case)
    steps = " ".join(out["next_steps"]).lower()

    assert "lab" in steps
    assert "imaging" in steps
    assert "vital" in steps


def test_agent_brain_confidence_gain_increases_with_more_modalities_and_posterior():
    agent = AgentBrain()

    case_min = _make_case(with_nlp=False, with_labs=False, with_imaging=False, with_vitals=False, with_posterior=False)
    gain_min = agent.reason(case_min)["confidence_gain"]

    case_more = _make_case(with_nlp=True, with_labs=True, with_imaging=True, with_vitals=True, with_posterior=True)
    gain_more = agent.reason(case_more)["confidence_gain"]

    assert 0.0 <= gain_min <= 1.0
    assert 0.0 <= gain_more <= 1.0
    assert gain_more > gain_min


def test_agent_brain_reasoning_mentions_modalities():
    agent = AgentBrain()
    case = _make_case(with_nlp=True, with_labs=True, with_imaging=False, with_vitals=False)

    reasoning = agent.reason(case)["agent_reasoning"].lower()

    assert "symptom" in reasoning or "nlp" in reasoning
    assert "labs" in reasoning
