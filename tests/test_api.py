"""
API integration tests.

`app.api.main.get_rag_chain` is monkeypatched to return a chain built on fake
embeddings/cross-encoder (see conftest.py), so these tests exercise the real
FastAPI routing, validation, and error-handling logic without needing network
access to download models.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.main as api_main
from app.chains.rag_chain import RAGChain
from app.reranker.cross_encoder_reranker import CrossEncoderReranker
from app.retriever.retriever import HybridRetriever


@pytest.fixture()
def client(sample_chunks, fake_vector_store, fake_cross_encoder, monkeypatch):
    retriever = HybridRetriever(fake_vector_store, sample_chunks)
    reranker = CrossEncoderReranker()
    chain = RAGChain(retriever, reranker)

    monkeypatch.setattr(api_main, "get_rag_chain", lambda force_reload=False: chain)
    monkeypatch.setattr(api_main, "load_chunks", lambda: sample_chunks)

    return TestClient(api_main.app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_endpoint_returns_grounded_answer(client):
    resp = client.post(
        "/query",
        json={"question": "What is the mileage of the diesel variant?", "car_brand": "Hyundai", "car_model": "Creta"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert len(body["sources"]) > 0


def test_query_endpoint_rejects_empty_question(client):
    resp = client.post("/query", json={"question": "   "})
    assert resp.status_code == 400


def test_search_endpoint_returns_raw_chunks(client):
    resp = client.post("/search", json={"question": "mileage", "car_brand": "Tata", "car_model": "Nexon"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    for chunk in body:
        assert chunk["metadata"]["car_brand"] == "Tata"


def test_chunks_endpoint_filters_by_brand(client):
    resp = client.get("/chunks", params={"car_brand": "Maruti Suzuki", "limit": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert all(c["metadata"]["car_brand"] == "Maruti Suzuki" for c in body["chunks"])


def test_metadata_endpoint_lists_facets(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert "Hyundai" in body["car_brands"]
    assert "car_models" in body
    assert "sections" in body


def test_upload_rejects_non_pdf(client):
    resp = client.post("/upload", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert resp.status_code == 400
