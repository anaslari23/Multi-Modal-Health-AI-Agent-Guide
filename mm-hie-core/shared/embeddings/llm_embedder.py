import os
from pathlib import Path
from typing import List, Optional

try:  # ImportError-safe import
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - simple import guard
    Llama = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DEFAULT_MODELS_DIR_ENV = "MODELS_DIR"
DEFAULT_RELATIVE_MODEL_PATH = Path("models") / "llm" / "gemma2_medquad.gguf"


class GGUFEmbedder:
    """Wrapper around llama-cpp-python for GGUF-based embeddings.

    Usage:
        from shared.embeddings.llm_embedder import GGUFEmbedder

        embedder = GGUFEmbedder()
        vec = embedder.embed("Short clinical text...")
        vecs = embedder.embed_batch(["fever", "dyspnea", "chest pain"])

    Notes:
        - Uses llama_cpp.Llama with CPU-safe defaults:
            n_ctx=4096, embedding=True, n_gpu_layers=0
        - Model path can be provided explicitly or resolved via:
            $MODELS_DIR/models/llm/gemma2_medquad.gguf
              or
            <repo_root>/models/llm/gemma2_medquad.gguf

        TODO (integration points):
        - ingestion/*_ingestor.py should call GGUFEmbedder to embed text
          before writing canonical JSON.
        - fusion/make_fusion_dataset.py can call embed() to generate
          NLP embeddings.
        - retrieval/build_pubmed_index.py should use embed_batch().
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with "
                "'python3 -m pip install llama-cpp-python' inside your virtualenv."
            ) from _IMPORT_ERROR

        resolved_model_path = self._resolve_model_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"GGUF model file not found at: {resolved_model_path}. "
                "Expected Gemma-2 MedQuad GGUF at this path. "
                "You may need to download it from 'mradermacher/Gemma-2-2b-it-chat-MedQuad-GGUF' "
                "and place it at MODELS_DIR/models/llm/gemma2_medquad.gguf."
            )

        # TODO: optionally implement automatic download if the file is missing.
        # For example, using huggingface_hub to fetch
        # 'mradermacher/Gemma-2-2b-it-chat-MedQuad-GGUF'.

        self._model_path = resolved_model_path
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=4096,
            embedding=True,
            n_gpu_layers=0,
        )

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Path:
        """Resolve the GGUF model path with sensible defaults.

        Priority:
        1) Explicit model_path argument.
        2) $MODELS_DIR/models/llm/gemma2_medquad.gguf
        3) <repo_root>/models/llm/gemma2_medquad.gguf
        """

        if model_path is not None:
            return Path(model_path).expanduser().resolve()

        # 2) MODELS_DIR env var, if provided
        models_dir_env = os.getenv(DEFAULT_MODELS_DIR_ENV)
        if models_dir_env:
            return Path(models_dir_env).expanduser().resolve() / DEFAULT_RELATIVE_MODEL_PATH

        # 3) Infer repo root from this file location: mm-hie-core/shared/embeddings/...
        this_file = Path(__file__).resolve()
        # repo_root / mm-hie-core / shared / embeddings / llm_embedder.py
        repo_root = this_file.parents[3]
        return repo_root / DEFAULT_RELATIVE_MODEL_PATH

    def embed(self, text: str) -> List[float]:
        """Compute an embedding for a single text.

        Returns a 1D list[float] vector.
        """

        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text)!r}")

        result = self._llm.create_embedding(text)
        data = result.get("data")
        if not data:
            raise RuntimeError("llama-cpp returned no embedding data for single text.")

        return data[0]["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts.

        Returns a list of embedding vectors, one per input text.
        """

        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list[str], got {type(texts)!r}")
        if not texts:
            return []
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("All items in texts must be str.")

        result = self._llm.create_embedding(texts)
        data = result.get("data")
        if not data or len(data) != len(texts):
            raise RuntimeError(
                "llama-cpp returned an unexpected number of embeddings: "
                f"expected {len(texts)}, got {len(data) if data is not None else 0}."
            )

        return [item["embedding"] for item in data]
