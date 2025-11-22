from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def flatten_modalities_to_text(example: Dict[str, Any]) -> Tuple[str, str]:
    """Convert a multi-modal JSON example into (input, output) text.

    Expected keys (all optional, best-effort):
      - symptoms: List[str]
      - labs: Dict[str, Any]
      - imaging: str or Dict
      - vitals: Dict[str, Any]
      - history: str
      - reasoning: target reasoning steps / summary

    Returns:
      input_text, output_text
    """

    symptoms = example.get("symptoms") or []
    if isinstance(symptoms, str):
        symptoms_str = symptoms
    else:
        symptoms_str = ", ".join(symptoms)

    labs = example.get("labs") or {}
    if isinstance(labs, dict):
        labs_str = "; ".join(f"{k}: {v}" for k, v in labs.items())
    else:
        labs_str = str(labs)

    imaging = example.get("imaging") or ""
    imaging_str = imaging if isinstance(imaging, str) else json.dumps(imaging)

    vitals = example.get("vitals") or {}
    if isinstance(vitals, dict):
        vitals_str = "; ".join(f"{k}: {v}" for k, v in vitals.items())
    else:
        vitals_str = str(vitals)

    history = str(example.get("history") or "")

    input_chunks: List[str] = []
    if symptoms_str:
        input_chunks.append(f"Symptoms: {symptoms_str}.")
    if labs_str:
        input_chunks.append(f"Labs: {labs_str}.")
    if imaging_str:
        input_chunks.append(f"Imaging: {imaging_str}.")
    if vitals_str:
        input_chunks.append(f"Vitals: {vitals_str}.")
    if history:
        input_chunks.append(f"History: {history}.")

    input_text = " \n".join(input_chunks)

    # Target reasoning steps / plan; fall back to generic summary if missing
    output_text = str(
        example.get("reasoning")
        or example.get("agent_summary")
        or example.get("target_text")
        or "Provide step-by-step reasoning for the clinical situation above."
    )

    return input_text, output_text


def convert_json_to_agent_jsonl(src_path: str, out_path: str) -> None:
    """Convert a JSON/JSONL of modality records into agent training JSONL.

    Output format: one JSON object per line with keys:
      - input: flattened prompt
      - output: reasoning steps / target text
    """

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(src_path)

    records: Iterable[Dict[str, Any]]
    if src.suffix.lower() == ".jsonl":
        records = (json.loads(line) for line in src.read_text().splitlines() if line.strip())
    else:
        # assume plain JSON list
        records = json.loads(src.read_text())

    out_lines: List[str] = []
    for rec in records:
        inp, out = flatten_modalities_to_text(rec)
        out_lines.append(json.dumps({"input": inp, "output": out}))

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(out_lines), encoding="utf-8")


def generate_llm_augmented_data_stub() -> None:
    """Placeholder for an LLM-in-the-loop data generator.

    This is intentionally a stub. In practice, you would:
      - Load base (input, output) pairs.
      - Call an external LLM (e.g., via OpenAI / local model) to generate
        additional reasoning variations.
      - Save augmented JSONL in the same {"input", "output"} format.

    External API calls and keys are not implemented here for security reasons.
    """

    print(
        "LLM-in-the-loop data generation is not implemented in this stub. "
        "Use your preferred LLM client to expand the dataset into the same JSONL schema."
    )
