import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.medquad_ingestor import MedQuadIngestor


def test_medquad_ingestor_adds_text_embedding(tmp_path: Path) -> None:
    """Smoke test that MedQuAD ingestor populates embeddings['text'] in episodes.

    This does not assert vector shape or values, only that the key exists and
    the field is JSON-serializable.
    """

    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    ingestor = MedQuadIngestor(raw_root=raw_root, processed_root=processed_root, split="train", max_episodes=1)

    # This will hit the real HF dataset. If that is undesirable in CI, replace
    # with a mocked dataset in a more detailed test.
    ingestor.download()
    samples = list(ingestor.transform())
    ingestor.canonicalize(samples)

    assert ingestor.episodes, "Expected at least one canonical episode"
    ep = ingestor.episodes[0].to_dict()

    assert "embeddings" in ep
    assert isinstance(ep["embeddings"], dict)
    assert "text" in ep["embeddings"], "embeddings['text'] key should be present"

    # Ensure JSON serialization succeeds (episodes are written this way in the pipeline)
    json.dumps(ep)  # should not raise
