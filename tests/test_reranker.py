"""Tests for app.reranker.cross_encoder_reranker.CrossEncoderReranker."""

from __future__ import annotations

from app.config.schemas import ChunkMetadata, RetrievedChunk
from app.reranker.cross_encoder_reranker import CrossEncoderReranker


def _make_chunk(text: str, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_name="test.pdf",
            car_brand="TestBrand",
            car_model="TestModel",
            page=1,
            section="general",
        ),
    )


def test_reranker_reorders_by_relevance(fake_cross_encoder):
    reranker = CrossEncoderReranker()
    candidates = [
        _make_chunk("The car has a spacious boot and comfortable seats.", "c1"),
        _make_chunk("Mileage of the diesel variant is 21 km/l on the highway.", "c2"),
        _make_chunk("The infotainment system has a 10 inch touchscreen.", "c3"),
    ]
    results = reranker.rerank("What is the mileage?", candidates, top_n=3)
    assert results[0].metadata.chunk_id == "c2"


def test_reranker_respects_top_n(fake_cross_encoder):
    reranker = CrossEncoderReranker()
    candidates = [_make_chunk(f"chunk text number {i} about cars", f"c{i}") for i in range(10)]
    results = reranker.rerank("cars", candidates, top_n=3)
    assert len(results) == 3


def test_reranker_empty_candidates_returns_empty_list(fake_cross_encoder):
    reranker = CrossEncoderReranker()
    assert reranker.rerank("anything", [], top_n=5) == []


def test_reranker_populates_rerank_score(fake_cross_encoder):
    reranker = CrossEncoderReranker()
    candidates = [_make_chunk("diesel mileage engine", "c1")]
    results = reranker.rerank("mileage", candidates, top_n=1)
    assert results[0].rerank_score is not None
