"""Tests for app.retriever.retriever.HybridRetriever."""

from __future__ import annotations

from app.config.schemas import MetadataFilter
from app.retriever.retriever import HybridRetriever


def test_retriever_respects_car_brand_and_model_filter(sample_chunks, fake_vector_store):
    retriever = HybridRetriever(fake_vector_store, sample_chunks)
    results = retriever.retrieve(
        "mileage",
        metadata_filter=MetadataFilter(car_brand="Hyundai", car_model="Creta"),
        top_k=10,
    )
    assert results, "Expected at least one result"
    for r in results:
        assert r.metadata.car_brand == "Hyundai"
        assert "creta" in r.metadata.car_model.lower()


def test_retriever_narrow_filter_returns_no_results_gracefully(sample_chunks, fake_vector_store):
    retriever = HybridRetriever(fake_vector_store, sample_chunks)
    results = retriever.retrieve(
        "mileage",
        metadata_filter=MetadataFilter(car_brand="Nonexistent Brand"),
        top_k=10,
    )
    assert results == []


def test_retriever_bm25_surfaces_exact_keyword_matches(sample_chunks, fake_vector_store):
    retriever = HybridRetriever(fake_vector_store, sample_chunks)
    results = retriever.retrieve("SX(O)", metadata_filter=None, top_k=15)
    assert any("SX(O)" in r.text for r in results)


def test_retriever_hybrid_score_is_populated(sample_chunks, fake_vector_store):
    retriever = HybridRetriever(fake_vector_store, sample_chunks)
    results = retriever.retrieve("engine displacement", top_k=5)
    assert results
    for r in results:
        assert r.hybrid_score is not None


def test_retriever_results_sorted_descending_by_hybrid_score(sample_chunks, fake_vector_store):
    retriever = HybridRetriever(fake_vector_store, sample_chunks)
    results = retriever.retrieve("safety airbags ADAS", top_k=10)
    scores = [r.hybrid_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_respects_section_filter(sample_chunks, fake_vector_store):
    retriever = HybridRetriever(fake_vector_store, sample_chunks)
    results = retriever.retrieve(
        "specifications", metadata_filter=MetadataFilter(section="dimensions"), top_k=20
    )
    for r in results:
        assert r.metadata.section == "dimensions"
