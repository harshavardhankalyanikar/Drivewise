"""Tests for app.ingestion.chunker."""

from __future__ import annotations

import pytest

from app.config.schemas import RawPageElement
from app.ingestion.chunker import (
    _classify_section,
    _detect_fuel_type,
    _detect_variant,
    _split_long_text,
    chunk_document,
)
from app.ingestion.pdf_parser import ParsedDocument, parse_pdf
from app.config.settings import settings

BROCHURES_DIR = settings.brochures_dir


def _skip_if_no_brochures():
    if not any(BROCHURES_DIR.glob("*.pdf")):
        pytest.skip("Run scripts/generate_sample_brochures.py first.")


def test_classify_section_engine():
    section = _classify_section("Engine and Performance", "1.5L turbo petrol engine, 160 PS")
    assert section.value == "engine_and_performance"


def test_classify_section_safety():
    section = _classify_section("Safety Features", "6 airbags, ADAS, ESC standard")
    assert section.value == "safety_features"


def test_classify_section_falls_back_to_general():
    section = _classify_section(None, "Lorem ipsum dolor sit amet with nothing relevant here.")
    assert section.value == "general"


def test_detect_fuel_type_diesel():
    assert _detect_fuel_type("The 1.5L CRDi diesel engine offers strong torque.") == "diesel"


def test_detect_fuel_type_turbo_petrol_preferred_over_petrol():
    assert _detect_fuel_type("The turbo-petrol variant produces 160 PS.") == "turbo_petrol"


def test_detect_fuel_type_ambiguous_returns_none():
    assert _detect_fuel_type("Both petrol and diesel variants are offered.") is None


def test_detect_variant_single_match():
    variants = ["E", "EX", "S", "SX", "SX(O)"]
    assert _detect_variant("Available only on the SX(O) trim.", variants) == "SX(O)"


def test_detect_variant_longer_name_not_shadowed_by_substring():
    variants = ["SX", "SX(O)"]
    result = _detect_variant("Exclusive to SX(O).", variants)
    assert result == "SX(O)"


def test_detect_variant_multiple_matches_returns_none():
    variants = ["S", "SX", "SX(O)"]
    assert _detect_variant("Available on S, SX and SX(O).", variants) is None


def test_split_long_text_respects_max_chars():
    long_text = " ".join([f"Sentence number {i}." for i in range(200)])
    pieces = _split_long_text(long_text, max_chars=200, overlap=20)
    assert len(pieces) > 1
    assert all(len(p) <= 260 for p in pieces)  # allow overlap slack


def test_split_long_text_short_text_returns_single_piece():
    text = "This is a short sentence."
    assert _split_long_text(text, max_chars=1200, overlap=150) == [text]


def test_chunk_document_produces_metadata_rich_chunks():
    _skip_if_no_brochures()
    doc = parse_pdf(BROCHURES_DIR / "hyundai_creta_2026.pdf")
    chunks = chunk_document(doc)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata.car_brand == "Hyundai"
        assert chunk.metadata.document_name == "hyundai_creta_2026.pdf"
        assert chunk.metadata.chunk_id
        assert chunk.metadata.page >= 1
        assert chunk.metadata.section


def test_chunk_document_tables_kept_intact():
    _skip_if_no_brochures()
    doc = parse_pdf(BROCHURES_DIR / "hyundai_creta_2026.pdf")
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if c.metadata.content_type == "table"]
    assert table_chunks, "Expected at least one table chunk"
    for tc in table_chunks:
        assert "|" in tc.text  # our row format uses " | " separators


def test_chunk_document_skips_empty_document():
    empty_doc = ParsedDocument("empty.pdf", "Unknown Brand", "Unknown Model", [])
    assert chunk_document(empty_doc) == []


def test_chunk_document_short_paragraphs_are_dropped():
    doc = ParsedDocument(
        "test.pdf",
        "TestBrand",
        "TestModel",
        [
            RawPageElement(page_number=1, element_type="heading", text="Engine and Performance", heading_level=1),
            RawPageElement(page_number=1, element_type="paragraph", text="Too short"),
        ],
    )
    chunks = chunk_document(doc)
    assert chunks == []  # below settings.min_chunk_chars
