from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .embedder import build_embedder
from .prompts import (
    DIAGNOSIS_PROMPT,
    TREATMENT_PROMPT,
    MEDICAL_EXPLAIN_PROMPT,
    DRUG_INFO_PROMPT,
)
from .retrievers import HybridRetriever
from .vector_store import VectorStore
from ..llm.reasoning_model import medical_reasoner


DEFAULT_CORPUS_PATH = Path("app/rag/datasets/corpus/merged_corpus.jsonl")
DEFAULT_INDEX_PATH = Path("app/rag/datasets/corpus/vector_index")


class RAGEngine:
    """Lightweight RAG engine for clinical reasoning.

    - Uses `medical_reasoner` (Ollama/LiteLLM) for generation.
    - Uses sentence-transformers embeddings + FAISS + BM25 for retrieval.
    """

    def __init__(
        self,
        corpus_path: Union[Path, str] = DEFAULT_CORPUS_PATH,
        index_path: Union[Path, str] = DEFAULT_INDEX_PATH,
        prefer_jina_embeddings: bool = False,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path)

        self._embedder = build_embedder(prefer_jina=prefer_jina_embeddings)
        self._vector_store: Optional[VectorStore] = None
        self._retriever: Optional[HybridRetriever] = None
        self._corpus: List[Dict[str, Any]] = []

    # --- Lazy loading --------------------------------------------------------

    def _load_corpus(self) -> None:
        if self._corpus:
            return
        if not self.corpus_path.exists():
            # No corpus yet; RAG will still run but with empty context.
            self._corpus = []
            return
        
        print(f"Loading corpus from {self.corpus_path}...")
        corpus: List[Dict[str, Any]] = []
        try:
            with self.corpus_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        corpus.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(corpus)} documents.")
        except Exception as e:
            print(f"Failed to load corpus: {e}")
            corpus = []
            
        self._corpus = corpus

    def _ensure_vector_store_and_retriever(self) -> None:
        if self._retriever is not None:
            return

        self._load_corpus()
        texts = [c.get("content", "") for c in self._corpus]
        if not texts:
            # Empty retriever that always returns no context.
            self._vector_store = VectorStore(dim=384)
            self._retriever = HybridRetriever([], [], self._vector_store)
            return

        # Try loading an existing FAISS index; otherwise build from scratch.
        dim = 384
        if self.index_path.with_suffix(".index").exists():
            vs = VectorStore.load_store(self.index_path, dim=dim)
        else:
            emb = self._embedder.embed_texts(texts)
            # Enrich metadata with text and index for retriever.
            metadatas = []
            for idx, c in enumerate(self._corpus):
                meta = dict(c)
                meta["_idx"] = idx
                metadatas.append(meta)
            vs = VectorStore(dim=emb.shape[1])
            vs.build_store(emb, metadatas)
            vs.save_store(self.index_path)

        self._vector_store = vs
        corpus_metadata = []
        for idx, c in enumerate(self._corpus):
            meta = dict(c)
            meta["_idx"] = idx
            corpus_metadata.append(meta)
        self._retriever = HybridRetriever(texts, corpus_metadata, vs)

    # --- Core helpers --------------------------------------------------------

    def _retrieved_context(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        self._ensure_vector_store_and_retriever()
        assert self._retriever is not None

        if not query:
            return []

        q_emb = self._embedder.embed_texts([query])[0]
        chunks = self._retriever.retrieve(query=query, query_vector=q_emb, top_k=top_k)
        return [c.metadata for c in chunks]

    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        parts = []
        for c in contexts:
            source = c.get("source", "unknown")
            title = c.get("title", "")
            content = c.get("content", "")
            parts.append(f"[source={source}] {title}\n{content}")
        return "\n\n".join(parts)

    def _llm_generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Use the shared medical_reasoner instance (Ollama/LiteLLM)
        try:
            return medical_reasoner.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,  # Lower temperature for more factual responses
            )
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return "I apologize, but I am unable to generate a response at this time due to a technical issue."

    # --- Public API ----------------------------------------------------------

    def query(
        self,
        question: Optional[str] = None,
        symptoms: Optional[str] = None,
        patient_info: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        q = question or symptoms or ""
        ctx_items = self._retrieved_context(q, top_k=top_k)
        ctx_str = self._format_context(ctx_items)

        prompt = f"""### Question:
{q}

### Retrieved Medical Knowledge:
{ctx_str}

### Task:
Provide a concise, clinically safe answer based ONLY on the Retrieved Medical Knowledge above.
If the answer is not in the context, say "I do not have enough information to answer this."
IMPORTANT: Cite sources using [Source: Title] format.
Do NOT hallucinate medicines or facts not present in the context.
"""
        answer = self._llm_generate(prompt)
        return {"answer": answer, "context": ctx_items}

    def diagnose(
        self,
        symptoms: str,
        patient_info: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        patient_info_str = patient_info or ""
        ctx_items = self._retrieved_context(symptoms, top_k=top_k)
        ctx_str = self._format_context(ctx_items)

        prompt = DIAGNOSIS_PROMPT.format(
            patient_info=patient_info_str,
            symptoms=symptoms,
            context=ctx_str,
        )
        answer = self._llm_generate(prompt)
        return {"answer": answer, "context": ctx_items}

    def explain(
        self,
        question: str,
        symptoms: Optional[str] = None,
        patient_info: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        q = question
        patient_info_str = patient_info or ""
        symptoms_str = symptoms or q
        ctx_items = self._retrieved_context(q, top_k=top_k)
        ctx_str = self._format_context(ctx_items)

        prompt = MEDICAL_EXPLAIN_PROMPT.format(
            patient_info=patient_info_str,
            symptoms=symptoms_str,
            context=ctx_str,
        )
        answer = self._llm_generate(prompt)
        return {"answer": answer, "context": ctx_items}

    def treatment(
        self,
        question: str,
        symptoms: Optional[str] = None,
        patient_info: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        q = question
        patient_info_str = patient_info or ""
        symptoms_str = symptoms or q
        ctx_items = self._retrieved_context(symptoms_str, top_k=top_k)
        ctx_str = self._format_context(ctx_items)

        prompt = TREATMENT_PROMPT.format(
            patient_info=patient_info_str,
            symptoms=symptoms_str,
            context=ctx_str,
        )
        answer = self._llm_generate(prompt)
        return {"answer": answer, "context": ctx_items}

    def drug_info(
        self,
        question: str,
        patient_info: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        patient_info_str = patient_info or ""
        ctx_items = self._retrieved_context(question, top_k=top_k)
        ctx_str = self._format_context(ctx_items)

        prompt = DRUG_INFO_PROMPT.format(
            patient_info=patient_info_str,
            symptoms=question,
            context=ctx_str,
        )
        answer = self._llm_generate(prompt)
        return {"answer": answer, "context": ctx_items}


# Singleton RAG engine used by API routes and orchestrator.
# Lazy initialization to avoid blocking server startup with vector index build
_rag_instance: Optional[RAGEngine] = None

def get_rag() -> RAGEngine:
    """Get or create the RAG engine instance (lazy initialization)."""
    global _rag_instance
    if _rag_instance is None:
        print("Initializing RAG engine (this may take a moment on first use)...")
        _rag_instance = RAGEngine()
    return _rag_instance

# Expose as 'rag' for backward compatibility - but it's now a function call
rag = get_rag
