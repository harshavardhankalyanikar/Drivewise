"""Tests for app.chains.rag_chain.RAGChain (using the offline 'template' LLM provider)."""

from __future__ import annotations

from app.config.schemas import QueryRequest
from app.chains.rag_chain import RAGChain
from app.prompts.templates import NO_INFO_SENTENCE
from app.reranker.cross_encoder_reranker import CrossEncoderReranker
from app.retriever.retriever import HybridRetriever


def _build_chain(sample_chunks, fake_vector_store, fake_cross_encoder):
    retriever = HybridRetriever(fake_vector_store, sample_chunks)
    reranker = CrossEncoderReranker()
    return RAGChain(retriever, reranker)


def test_chain_answers_grounded_question(sample_chunks, fake_vector_store, fake_cross_encoder):
    chain = _build_chain(sample_chunks, fake_vector_store, fake_cross_encoder)
    response = chain.answer(
        QueryRequest(question="What is the mileage of the diesel variant?", car_brand="Hyundai", car_model="Creta")
    )
    assert response.grounded is True
    assert response.sources
    assert "km/l" in response.answer or "km/l" in " ".join(s.snippet for s in response.sources)


def test_chain_returns_no_info_sentence_for_impossible_filter(sample_chunks, fake_vector_store, fake_cross_encoder):
    chain = _build_chain(sample_chunks, fake_vector_store, fake_cross_encoder)
    response = chain.answer(
        QueryRequest(question="What is the mileage?", car_brand="NonexistentBrand12345")
    )
    assert response.grounded is False
    assert response.answer == NO_INFO_SENTENCE
    assert response.sources == []


def test_chain_confidence_is_zero_when_not_grounded(sample_chunks, fake_vector_store, fake_cross_encoder):
    chain = _build_chain(sample_chunks, fake_vector_store, fake_cross_encoder)
    response = chain.answer(QueryRequest(question="xyz", car_brand="NonexistentBrand12345"))
    assert response.confidence == 0.0


def test_chain_metadata_filters_applied_reflects_request(sample_chunks, fake_vector_store, fake_cross_encoder):
    chain = _build_chain(sample_chunks, fake_vector_store, fake_cross_encoder)
    response = chain.answer(QueryRequest(question="sunroof", car_brand="Tata", car_model="Nexon"))
    assert response.metadata_filters_applied.get("car_brand") == "Tata"
    assert response.metadata_filters_applied.get("car_model") == "Nexon"


def test_chain_reasoning_summary_mentions_sections(sample_chunks, fake_vector_store, fake_cross_encoder):
    chain = _build_chain(sample_chunks, fake_vector_store, fake_cross_encoder)
    response = chain.answer(QueryRequest(question="airbags safety rating", car_brand="Maruti Suzuki"))
    assert "chunk" in response.reasoning_summary.lower()
