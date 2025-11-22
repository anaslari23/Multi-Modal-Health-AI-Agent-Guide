from typing import Iterable, List

import numpy as np

try:  # Optional dependency; server should still start without it
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # noqa: BLE001
    SentenceTransformer = None  # type: ignore


class Embedder:
    """CPU-only text embedder with pluggable sentence-transformers models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        # Device="cpu" keeps things Mac/MPS safe; sentence-transformers will
        # internally use torch with no CUDA.
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name, device="cpu")
        else:
            self.model = None

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = [t if t is not None else "" for t in texts]
        if not texts_list:
            return np.zeros((0, 384), dtype="float32")
        if self.model is None:
            # Fallback: return zero embeddings with MiniLM dim 384.
            return np.zeros((len(texts_list), 384), dtype="float32")
        emb = self.model.encode(texts_list, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype("float32")


def build_embedder(prefer_jina: bool = False) -> Embedder:
    """Factory returning an Embedder.

    prefer_jina=True will use jinaai/jina-embeddings-v2-base, which is still CPU-safe
    but heavier than MiniLM.
    """

    if prefer_jina:
        return Embedder("jinaai/jina-embeddings-v2-base")
    return Embedder("sentence-transformers/all-MiniLM-L6-v2")
