import argparse
import csv
import json
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


@dataclass
class CodeNode:
    code: str
    system: str  # e.g. "ICD-10", "SNOMED-CT", "RxNorm"
    term: str
    synonyms: List[str]
    parents: List[str]
    children: List[str]


class OntologyGraph:
    """Lightweight ICD-10 / SNOMED / RxNorm ontology graph.

    This is intentionally simple and file-backed so it can be regenerated
    offline from CSV mapping drops.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, CodeNode] = {}
        # String index for quick text lookup -> codes
        self._string_index: Dict[str, Set[str]] = {}
        # Map disease code -> set of RxNorm medication codes (indications)
        self.disease_to_meds: Dict[str, Set[str]] = {}

    # ---------------------- Loading helpers ----------------------
    def _add_node(
        self,
        code: str,
        system: str,
        term: str,
        synonyms: Optional[List[str]] = None,
        parents: Optional[List[str]] = None,
    ) -> None:
        key = f"{system}:{code}"
        if key in self.nodes:
            node = self.nodes[key]
            if term and term not in node.synonyms and term != node.term:
                node.synonyms.append(term)
            for s in synonyms or []:
                if s not in node.synonyms:
                    node.synonyms.append(s)
            for p in parents or []:
                if p not in node.parents:
                    node.parents.append(p)
        else:
            node = CodeNode(
                code=code,
                system=system,
                term=term,
                synonyms=list(synonyms or []),
                parents=list(parents or []),
                children=[],
            )
            self.nodes[key] = node

        # Update string index
        self._index_string(term, key)
        for s in node.synonyms:
            self._index_string(s, key)

    def _index_string(self, text: str, key: str) -> None:
        norm = self._normalize_text(text)
        if not norm:
            return
        self._string_index.setdefault(norm, set()).add(key)

    def _link_parent_child(self) -> None:
        for key, node in self.nodes.items():
            for parent_key in node.parents:
                parent = self.nodes.get(parent_key)
                if parent and key not in parent.children:
                    parent.children.append(key)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    # Public factory
    @classmethod
    def from_csvs(
        cls,
        icd10_to_snomed_csv: Path,
        rxnorm_mappings_csv: Path,
    ) -> "OntologyGraph":
        graph = cls()

        # Load ICD-10 <-> SNOMED mappings
        if icd10_to_snomed_csv.exists():
            with icd10_to_snomed_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    icd = row.get("icd10_code") or ""
                    icd_term = row.get("icd10_term") or ""
                    icd_synonyms = (row.get("icd10_synonyms") or "").split("|") if row.get("icd10_synonyms") else []

                    snomed = row.get("snomed_id") or ""
                    snomed_term = row.get("snomed_term") or ""
                    snomed_synonyms = (row.get("snomed_synonyms") or "").split("|") if row.get("snomed_synonyms") else []
                    parent_snomed = row.get("parent_snomed_id") or ""

                    if icd:
                        graph._add_node(
                            code=icd,
                            system="ICD-10",
                            term=icd_term or icd,
                            synonyms=icd_synonyms,
                            parents=[],
                        )
                    if snomed:
                        parents = []
                        if parent_snomed:
                            parents.append(f"SNOMED-CT:{parent_snomed}")
                        graph._add_node(
                            code=snomed,
                            system="SNOMED-CT",
                            term=snomed_term or snomed,
                            synonyms=snomed_synonyms,
                            parents=parents,
                        )

                    # Cross-system linkage can be represented via disease_to_meds
                    # or an external mapping; here we simply ensure both sides are present.

        else:
            logger.warning("ICD10->SNOMED CSV not found: %s", icd10_to_snomed_csv)

        # Load RxNorm mappings
        if rxnorm_mappings_csv.exists():
            with rxnorm_mappings_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rx = row.get("rxnorm_id") or ""
                    ingredient = row.get("ingredient") or ""
                    form = row.get("form") or ""
                    synonyms: List[str] = []
                    if ingredient:
                        synonyms.append(ingredient)
                    if form:
                        synonyms.append(form)

                    if rx:
                        graph._add_node(
                            code=rx,
                            system="RxNorm",
                            term=ingredient or rx,
                            synonyms=synonyms,
                            parents=[],
                        )

                    # Disease indication links
                    icd_ind = row.get("indication_icd10") or ""
                    snomed_ind = row.get("indication_snomed") or ""
                    if icd_ind:
                        disease_key = f"ICD-10:{icd_ind}"
                        graph.disease_to_meds.setdefault(disease_key, set()).add(f"RxNorm:{rx}")
                    if snomed_ind:
                        disease_key = f"SNOMED-CT:{snomed_ind}"
                        graph.disease_to_meds.setdefault(disease_key, set()).add(f"RxNorm:{rx}")
        else:
            logger.warning("RxNorm mappings CSV not found: %s", rxnorm_mappings_csv)

        graph._link_parent_child()
        return graph

    # ---------------------- Query helpers ----------------------
    def map_disease_string(self, text: str) -> List[str]:
        """Map a free-text disease string to candidate canonical codes.

        Returns a list of keys like "ICD-10:I50.9" or "SNOMED-CT:42343007".
        """
        norm = self._normalize_text(text)
        return sorted(self._string_index.get(norm, []))

    def get_synonyms(self, system: str, code: str) -> List[str]:
        key = f"{system}:{code}"
        node = self.nodes.get(key)
        if not node:
            return []
        return [node.term] + node.synonyms

    def get_parents(self, system: str, code: str) -> List[CodeNode]:
        key = f"{system}:{code}"
        node = self.nodes.get(key)
        if not node:
            return []
        return [self.nodes[p] for p in node.parents if p in self.nodes]

    def get_children(self, system: str, code: str) -> List[CodeNode]:
        key = f"{system}:{code}"
        node = self.nodes.get(key)
        if not node:
            return []
        return [self.nodes[c] for c in node.children if c in self.nodes]

    def get_meds_for_disease(self, system: str, code: str) -> List[CodeNode]:
        key = f"{system}:{code}"
        meds_keys = self.disease_to_meds.get(key, set())
        return [self.nodes[k] for k in meds_keys if k in self.nodes]

    # ---------------------- Serialization ----------------------
    def to_json_dict(self) -> Dict[str, Dict]:
        return {
            key: {
                "code": node.code,
                "system": node.system,
                "term": node.term,
                "synonyms": node.synonyms,
                "parents": node.parents,
                "children": node.children,
            }
            for key, node in self.nodes.items()
        }

    # Convenience helper functions


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def build_ontology(
    icd10_to_snomed_csv: Path,
    rxnorm_mappings_csv: Path,
    output_json: Path,
    output_pkl: Path,
) -> OntologyGraph:
    logger.info("Building ontology from %s and %s", icd10_to_snomed_csv, rxnorm_mappings_csv)
    graph = OntologyGraph.from_csvs(icd10_to_snomed_csv, rxnorm_mappings_csv)

    # Export JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f_json:
        json.dump(graph.to_json_dict(), f_json, ensure_ascii=False, indent=2)
    logger.info("Wrote ontology JSON to %s", output_json)

    # Export pickle
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f_pkl:
        pickle.dump(graph, f_pkl)
    logger.info("Wrote ontology pickle to %s", output_pkl)

    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ICD-10 / SNOMED / RxNorm ontology")
    parser.add_argument(
        "--icd10-to-snomed",
        type=Path,
        default=Path("mm-hie-scale/ontology/mappings/icd10_to_snomed.csv"),
        help="CSV with ICD-10 to SNOMED mappings",
    )
    parser.add_argument(
        "--rxnorm-mappings",
        type=Path,
        default=Path("mm-hie-scale/ontology/mappings/rxnorm_mappings.csv"),
        help="CSV with RxNorm medication mappings",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("mm-hie-scale/ontology/ontology.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=Path("mm-hie-scale/ontology/ontology.pkl"),
        help="Output pickle file path",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        build_ontology(
            icd10_to_snomed_csv=args.icd10_to_snomed,
            rxnorm_mappings_csv=args.rxnorm_mappings,
            output_json=args.output_json,
            output_pkl=args.output_pkl,
        )
    except Exception:
        logger.exception("Failed to build ontology")
        raise


if __name__ == "__main__":
    main()
