import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class CanonicalEpisode:
    episode_id: str
    notes: Optional[List[Dict[str, Any]]] = None
    labs: Optional[List[Dict[str, Any]]] = None
    vitals: Optional[List[Dict[str, Any]]] = None
    imaging: Optional[List[Dict[str, Any]]] = None
    labels: Optional[List[Dict[str, Any]]] = None
    timestamps: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def deidentify_record(record: Dict[str, Any], phi_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Simple de-identification stub removing common PHI-style keys.

    This is intentionally conservative and schema-agnostic; extend as needed
    for each dataset (e.g. drop exact DOB, address, MRN, free-text identifiers).
    """
    if phi_keys is None:
        phi_keys = [
            "name",
            "patient_name",
            "dob",
            "date_of_birth",
            "address",
            "mrn",
            "medical_record_number",
            "phone",
            "email",
        ]

    cleaned = {}
    for k, v in record.items():
        if k.lower() in phi_keys:
            continue
        cleaned[k] = v
    return cleaned


def ensure_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create directory: %s", path)
        raise


def build_episode(
    episode_id: str,
    notes: Optional[List[Dict[str, Any]]] = None,
    labs: Optional[List[Dict[str, Any]]] = None,
    vitals: Optional[List[Dict[str, Any]]] = None,
    imaging: Optional[List[Dict[str, Any]]] = None,
    labels: Optional[List[Dict[str, Any]]] = None,
    timestamps: Optional[Dict[str, Any]] = None,
) -> CanonicalEpisode:
    return CanonicalEpisode(
        episode_id=episode_id,
        notes=notes or [],
        labs=labs or [],
        vitals=vitals or [],
        imaging=imaging or [],
        labels=labels or [],
        timestamps=timestamps or {},
    )


def write_episode_json(episode: CanonicalEpisode, output_dir: Path) -> Path:
    """Write one canonical episode as pretty JSON to output_dir.

    File name convention: <episode_id>.json
    """
    import json

    ensure_directory(output_dir)
    output_path = output_dir / f"{episode.episode_id}.json"
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(episode.to_dict(), f, ensure_ascii=False, indent=2)
        logger.debug("Wrote episode %s to %s", episode.episode_id, output_path)
    except Exception:
        logger.exception("Failed to write episode %s to %s", episode.episode_id, output_path)
        raise

    return output_path
