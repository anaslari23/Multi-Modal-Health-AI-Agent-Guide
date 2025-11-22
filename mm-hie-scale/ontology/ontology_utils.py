import functools
import logging
import pickle
from pathlib import Path
from typing import List, Optional

from .build_ontology import OntologyGraph


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_ontology(path: Optional[Path] = None) -> OntologyGraph:
    """Load ontology.pkl once and cache it.

    By default looks for mm-hie-scale/ontology/ontology.pkl relative to CWD.
    """
    if path is None:
        path = Path("mm-hie-scale/ontology/ontology.pkl")

    if not path.exists():
        raise FileNotFoundError(f"Ontology pickle not found at {path}. Run build_ontology.py first.")

    logger.info("Loading ontology from %s", path)
    with path.open("rb") as f:
        graph: OntologyGraph = pickle.load(f)
    return graph


def normalize_disease(text: str, ontology_path: Optional[Path] = None) -> List[str]:
    """Map a free-text disease string to canonical codes.

    Returns list of keys like ["ICD-10:I50.9", "SNOMED-CT:42343007"].
    """
    graph = _load_ontology(ontology_path)
    return graph.map_disease_string(text)


def get_disease_synonyms(system: str, code: str, ontology_path: Optional[Path] = None) -> List[str]:
    """Return preferred term + synonyms for a disease code."""
    graph = _load_ontology(ontology_path)
    return graph.get_synonyms(system, code)


def get_parent_diseases(system: str, code: str, ontology_path: Optional[Path] = None) -> List[dict]:
    """Return parent disease nodes as dicts.

    Each dict has keys: code, system, term, synonyms, parents, children.
    """
    graph = _load_ontology(ontology_path)
    parents = graph.get_parents(system, code)
    return [
        {
            "code": n.code,
            "system": n.system,
            "term": n.term,
            "synonyms": n.synonyms,
            "parents": n.parents,
            "children": n.children,
        }
        for n in parents
    ]


def get_child_diseases(system: str, code: str, ontology_path: Optional[Path] = None) -> List[dict]:
    """Return child disease nodes as dicts."""
    graph = _load_ontology(ontology_path)
    children = graph.get_children(system, code)
    return [
        {
            "code": n.code,
            "system": n.system,
            "term": n.term,
            "synonyms": n.synonyms,
            "parents": n.parents,
            "children": n.children,
        }
        for n in children
    ]


def get_meds_for_disease(system: str, code: str, ontology_path: Optional[Path] = None) -> List[dict]:
    """Return RxNorm medications indicated for this disease.

    Each dict has keys: code, system, term, synonyms, parents, children.
    """
    graph = _load_ontology(ontology_path)
    meds = graph.get_meds_for_disease(system, code)
    return [
        {
            "code": n.code,
            "system": n.system,
            "term": n.term,
            "synonyms": n.synonyms,
            "parents": n.parents,
            "children": n.children,
        }
        for n in meds
    ]
