"""
FAISS vector store layer.

Built on top of LangChain's `FAISS` vectorstore so the index can be persisted
to disk and reloaded without re-embedding. Metadata (car_brand, car_model,
variant, fuel_type, section, page, ...) is stored alongside every vector so
the retriever can pre-filter candidates before running similarity search.

Kept behind a small interface (`build`, `save`, `load`, `similarity_search`)
so swapping FAISS for ChromaDB later only requires a new class implementing
the same methods -- callers never touch FAISS internals directly.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS

from app.config.schemas import Chunk, MetadataFilter, RetrievedChunk, ChunkMetadata
from app.config.settings import settings
from app.embeddings.embedder import Embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _chunk_to_document(chunk: Chunk) -> Document:
    return Document(page_content=chunk.text, metadata=chunk.metadata.model_dump())


def _document_to_retrieved_chunk(doc: Document, score: float | None = None) -> RetrievedChunk:
    metadata = ChunkMetadata(**doc.metadata)
    return RetrievedChunk(text=doc.page_content, metadata=metadata, semantic_score=score)


class FaissVectorStore:
    """Wraps a LangChain FAISS index with persistence and metadata filtering."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self._store: FAISS | None = None

    # ------------------------------------------------------------------ build
    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a FAISS index from zero chunks.")
        documents = [_chunk_to_document(c) for c in chunks]
        logger.info("Building FAISS index from %d chunks...", len(documents))
        self._store = FAISS.from_documents(documents, self.embedder.langchain_embeddings)
        logger.info("FAISS index built with %d vectors.", self._store.index.ntotal)

    # ------------------------------------------------------------------ persist
    def save(self, path: Path | str | None = None) -> None:
        if self._store is None:
            raise RuntimeError("No index in memory to save. Call build() or load() first.")
        path = Path(path or settings.faiss_index_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(path))
        logger.info("Saved FAISS index to %s", path)

    def load(self, path: Path | str | None = None) -> None:
        path = Path(path or settings.faiss_index_dir)
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index found at {path}. Run the ingestion script first."
            )
        self._store = FAISS.load_local(
            str(path),
            self.embedder.langchain_embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index from %s (%d vectors)", path, self._store.index.ntotal)

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    # ------------------------------------------------------------------ search
    @staticmethod
    def _matches_filter(metadata: dict, metadata_filter: MetadataFilter) -> bool:
        checks = [
            (metadata_filter.car_brand, metadata.get("car_brand")),
            (metadata_filter.car_model, metadata.get("car_model")),
            (metadata_filter.variant, metadata.get("variant")),
            (metadata_filter.fuel_type, metadata.get("fuel_type")),
            (metadata_filter.section, metadata.get("section")),
            (metadata_filter.page, metadata.get("page")),
        ]
        for wanted, actual in checks:
            if wanted is None:
                continue
            if isinstance(wanted, str) and isinstance(actual, str):
                if wanted.strip().lower() != actual.strip().lower():
                    return False
            elif wanted != actual:
                return False
        return True

    def similarity_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        if self._store is None:
            raise RuntimeError("FAISS index not loaded. Call build() or load() first.")

        # LangChain FAISS supports a filter callable/dict; we over-fetch then
        # post-filter defensively for the (fairly complex) combined-field logic above.
        fetch_k = max(top_k * 6, 50)
        results = self._store.similarity_search_with_relevance_scores(query, k=fetch_k)

        filtered: list[RetrievedChunk] = []
        for doc, score in results:
            if metadata_filter and not self._matches_filter(doc.metadata, metadata_filter):
                continue
            filtered.append(_document_to_retrieved_chunk(doc, score))
            if len(filtered) >= top_k:
                break

        return filtered

    def all_documents(self) -> list[Document]:
        if self._store is None:
            raise RuntimeError("FAISS index not loaded.")
        return list(self._store.docstore._dict.values())  # noqa: SLF001 - LangChain has no public iterator
