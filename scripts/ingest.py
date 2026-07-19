"""
End-to-end ingestion script.

    python scripts/ingest.py [--brochures-dir PATH]

Runs: PDF parsing -> metadata-aware chunking -> embedding -> FAISS index
build -> persistence (both the FAISS index and a chunks.jsonl used to rebuild
the BM25 keyword index at query time without re-parsing PDFs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.embeddings.embedder import Embedder  # noqa: E402
from app.ingestion.chunker import chunk_documents  # noqa: E402
from app.ingestion.pdf_parser import parse_pdf_directory  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from app.utils.persistence import save_chunks  # noqa: E402
from app.vectorstore.faiss_store import FaissVectorStore  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest brochure PDFs into the DriveWise FAISS index.")
    parser.add_argument("--brochures-dir", type=str, default=str(settings.brochures_dir))
    parser.add_argument("--index-dir", type=str, default=str(settings.faiss_index_dir))
    args = parser.parse_args()

    brochures_dir = Path(args.brochures_dir)
    logger.info("Parsing PDFs from %s", brochures_dir)
    parsed_documents = parse_pdf_directory(brochures_dir)

    if not any(doc.elements for doc in parsed_documents):
        logger.error("No content extracted from any PDF in %s. Aborting.", brochures_dir)
        sys.exit(1)

    logger.info("Chunking %d parsed document(s)...", len(parsed_documents))
    chunks = chunk_documents(parsed_documents)
    logger.info("Produced %d total chunks.", len(chunks))

    chunks_path = save_chunks(chunks)
    logger.info("Persisted chunk metadata to %s", chunks_path)

    embedder = Embedder()
    store = FaissVectorStore(embedder=embedder)
    store.build(chunks)
    store.save(args.index_dir)

    logger.info("Ingestion complete. Index ready at %s", args.index_dir)


if __name__ == "__main__":
    main()
