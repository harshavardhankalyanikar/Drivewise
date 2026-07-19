"""
Embedding generation.

Wraps `sentence-transformers` (via LangChain's HuggingFaceEmbeddings) behind a
small, swappable interface. The concrete model is fully configurable through
`settings.embedding_model_name` (defaults to all-MiniLM-L6-v2; drop in
BAAI/bge-small-en-v1.5 or any other sentence-transformers checkpoint without
touching calling code).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _load_embeddings(model_name: str, device: str) -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model '%s' on device '%s'", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


class Embedder:
    """Thin, testable facade over the underlying embedding model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        self._model = _load_embeddings(self.model_name, self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    @property
    def langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Expose the raw LangChain embeddings object for FAISS.from_documents(...)."""
        return self._model


def get_default_embedder() -> Embedder:
    return Embedder()
