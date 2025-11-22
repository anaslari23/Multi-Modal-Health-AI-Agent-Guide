from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: List[Dict[str, Any]] = []

    # --- Build / persist -----------------------------------------------------

    def build_store(self, embeddings: np.ndarray, metadatas: Iterable[Dict[str, Any]]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        self.index.reset()
        self.index.add(embeddings)
        self.metadata = list(metadatas)

    def save_store(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".index")))
        with path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load_store(cls, path: Union[str, Path], dim: int) -> "VectorStore":
        path = Path(path)
        vs = cls(dim)
        vs.index = faiss.read_index(str(path.with_suffix(".index")))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                vs.metadata = json.load(f)
        else:
            vs.metadata = []
        return vs

    # --- Query ---------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if query_vector.ndim == 1:
            query_vector = query_vector[None, :]
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype("float32")

        distances, indices = self.index.search(query_vector, top_k)
        results: List[Tuple[float, Dict[str, Any]]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append((float(dist), self.metadata[idx]))
        return results
