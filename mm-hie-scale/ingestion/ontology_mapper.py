"""Thin wrappers and stubs around the ontology utilities.

This module centralizes ontology-related lookups for ingestion:
  - ICD10 -> SNOMED mapping
  - Disease free-text -> canonical codes
  - Drug name -> RxNorm codes

Dataset-specific ingestors should use these helpers (not the ontology
internals) so we can evolve the backing implementation without touching
all ingestors.
"""

from typing import List
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ontology.ontology_utils import normalize_disease


def map_disease_string_to_codes(text: str) -> List[str]:
    """Return canonical codes for a disease free-text string.

    Uses the prebuilt ontology graph; returns keys like
    ["ICD-10:I50.9", "SNOMED-CT:42343007"].
    """

    return normalize_disease(text)


def map_icd10_to_snomed(icd_code: str) -> List[str]:
    """Stub for ICD-10 -> SNOMED mapping.

    Currently delegates to the disease string mapping as a heuristic.
    You can replace this with a direct table lookup if desired.
    """

    keys = normalize_disease(icd_code)
    out: List[str] = []
    for key in keys:
        try:
            system, code = key.split(":", 1)
        except ValueError:
            continue
        if system == "SNOMED-CT":
            out.append(code)
    return sorted(set(out))


def map_drugname_to_rxnorm(drug_name: str) -> List[str]:
    """Stub for DrugName -> RxNorm mapping.

    TODO: Implement lookup via RxNorm mappings from the ontology graph
    or an external CSV. For now this returns an empty list so ingestors
    can call it safely without failing.
    """

    # TODO: integrate with ontology graph's medication nodes if available.
    _ = drug_name
    return []
