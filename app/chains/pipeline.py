"""
Singleton pipeline construction.

Both the FastAPI app and the Streamlit UI need the same fully-wired
retriever/reranker/chain, built once (loading the embedding model and FAISS
index is expensive). This module owns that lifecycle so neither caller has to
duplicate wiring logic.
"""

from __future__ import annotations

from app.chains.rag_chain import RAGChain
from app.embeddings.embedder import Embedder
from app.ingestion.chunker import chunk_documents
from app.ingestion.pdf_parser import parse_pdf_directory
from app.reranker.cross_encoder_reranker import CrossEncoderReranker
from app.retriever.retriever import HybridRetriever
from app.utils.logger import get_logger
from app.utils.persistence import load_chunks, save_chunks
from app.vectorstore.faiss_store import FaissVectorStore

logger = get_logger(__name__)

_chain: RAGChain | None = None


def get_rag_chain(force_reload: bool = False) -> RAGChain:
    """Return the process-wide RAGChain, building it on first call."""
    global _chain
    if _chain is not None and not force_reload:
        return _chain

    embedder = Embedder()
    store = FaissVectorStore(embedder=embedder)
    store.load()  # raises FileNotFoundError with a clear message if ingestion hasn't run yet

    chunks = load_chunks()
    retriever = HybridRetriever(store, chunks)
    reranker = CrossEncoderReranker()

    _chain = RAGChain(retriever, reranker)
    logger.info("RAGChain initialised with %d chunks.", len(chunks))
    return _chain


def rebuild_index_from_brochures(brochures_dir=None) -> int:
    """
    Re-run parsing -> chunking -> embedding -> FAISS build from the brochures
    directory (used by the /upload endpoint after a new PDF is added).
    Returns the number of chunks produced.
    """
    from app.config.settings import settings

    global _chain
    brochures_dir = brochures_dir or settings.brochures_dir

    parsed_documents = parse_pdf_directory(brochures_dir)
    chunks = chunk_documents(parsed_documents)
    if not chunks:
        raise ValueError(f"No chunks could be produced from PDFs in {brochures_dir}")

    save_chunks(chunks)

    embedder = Embedder()
    store = FaissVectorStore(embedder=embedder)
    store.build(chunks)
    store.save()

    _chain = None  # force re-construction with fresh index on next get_rag_chain()
    return len(chunks)
