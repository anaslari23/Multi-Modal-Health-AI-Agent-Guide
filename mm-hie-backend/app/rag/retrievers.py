from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    score: float
    metadata: Dict[str, Any]


class HybridRetriever:
    def __init__(
        self,
        corpus_texts: Iterable[str],
        corpus_metadata: Iterable[Dict[str, Any]],
        vector_store: VectorStore,
        bm25_weight: float = 0.5,
        embed_weight: float = 0.5,
    ) -> None:
        self.texts: List[str] = list(corpus_texts)
        self.metadata: List[Dict[str, Any]] = list(corpus_metadata)
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.embed_weight = embed_weight

        tokenized_corpus = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _bm25_scores(self, query: str, top_k: int) -> Dict[int, float]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        idx_scores = list(enumerate(scores))
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        return {idx: float(score) for idx, score in idx_scores[:top_k] if score > 0}

    def _emb_scores(self, query_vector: np.ndarray, top_k: int) -> Dict[int, float]:
        results = self.vector_store.search(query_vector, top_k)
        # VectorStore returns (distance, metadata), but we need indices.
        # Store index as part of metadata when building the store.
        scores: Dict[int, float] = {}
        for dist, meta in results:
            idx = int(meta.get("_idx", -1))
            if idx >= 0:
                # Convert L2 distance into a similarity-like score.
                scores[idx] = -float(dist)
        return scores

    def retrieve(self, query: str, query_vector: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        bm25_scores = self._bm25_scores(query, top_k * 4)
        emb_scores = self._emb_scores(query_vector, top_k * 4)

        all_indices = set(bm25_scores.keys()) | set(emb_scores.keys())
        fused: List[RetrievedChunk] = []
        for idx in all_indices:
            b = bm25_scores.get(idx, 0.0)
            e = emb_scores.get(idx, 0.0)
            score = self.bm25_weight * b + self.embed_weight * e
            fused.append(
                RetrievedChunk(
                    score=score,
                    metadata={**self.metadata[idx], "text": self.texts[idx]},
                )
            )

        fused.sort(key=lambda c: c.score, reverse=True)
        return fused[:top_k]
