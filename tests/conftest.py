"""
Shared pytest fixtures.

Tests use lightweight, deterministic fake stand-ins for the embedding model
and cross-encoder instead of downloading real weights from HuggingFace Hub.
This is standard practice for unit-testing ML pipelines: it keeps the test
suite fast, deterministic, and runnable in CI/offline environments, while the
real models are exercised via the ingestion script and API in an environment
with internet access. Pipeline *logic* (filtering, hybrid scoring, chunk
integrity, re-rank ordering, chain wiring) is fully exercised either way --
only the embedding/relevance *quality* differs.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from app.config.settings import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.pdf_parser import parse_pdf_directory
from app.vectorstore.faiss_store import FaissVectorStore


class DeterministicFakeEmbeddings(Embeddings):
    """Hash-seeded pseudo-embeddings: same text always yields the same vector."""

    def _vec(self, text: str) -> list[float]:
        seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        vec = np.array([rng.gauss(0, 1) for _ in range(64)])
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeEmbedder:
    """Duck-types app.embeddings.embedder.Embedder without loading a real model."""

    def __init__(self) -> None:
        self.langchain_embeddings = DeterministicFakeEmbeddings()


class FakeCrossEncoder:
    """Crude keyword-overlap relevance score, standing in for a real cross-encoder."""

    def predict(self, pairs):
        scores = []
        for query, doc in pairs:
            q_terms = set(query.lower().split())
            d_terms = set(doc.lower().split())
            scores.append(float(len(q_terms & d_terms)))
        return scores


@pytest.fixture(scope="session")
def sample_chunks():
    if not any(settings.brochures_dir.glob("*.pdf")):
        pytest.skip("Run scripts/generate_sample_brochures.py before running tests.")
    docs = parse_pdf_directory(settings.brochures_dir)
    return chunk_documents(docs)


@pytest.fixture(scope="session")
def fake_vector_store(sample_chunks):
    store = FaissVectorStore(embedder=FakeEmbedder())
    store.build(sample_chunks)
    return store


@pytest.fixture()
def fake_cross_encoder(monkeypatch):
    import app.reranker.cross_encoder_reranker as reranker_module

    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda name: FakeCrossEncoder())
    return FakeCrossEncoder()
