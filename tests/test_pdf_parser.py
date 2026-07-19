"""Tests for app.ingestion.pdf_parser using the generated sample brochures."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.pdf_parser import parse_pdf, parse_pdf_directory
from app.config.settings import settings

BROCHURES_DIR = settings.brochures_dir


def _skip_if_no_brochures():
    if not any(BROCHURES_DIR.glob("*.pdf")):
        pytest.skip(
            "No sample brochures found. Run "
            "`python scripts/generate_sample_brochures.py` first."
        )


def test_parse_pdf_directory_finds_all_brochures():
    _skip_if_no_brochures()
    docs = parse_pdf_directory(BROCHURES_DIR)
    assert len(docs) == len(list(BROCHURES_DIR.glob("*.pdf")))
    assert all(doc.elements for doc in docs)


def test_parse_pdf_infers_brand_and_model():
    _skip_if_no_brochures()
    doc = parse_pdf(BROCHURES_DIR / "hyundai_creta_2026.pdf")
    assert doc.car_brand == "Hyundai"
    assert "creta" in doc.car_model.lower()


def test_parse_pdf_extracts_headings_paragraphs_and_tables():
    _skip_if_no_brochures()
    doc = parse_pdf(BROCHURES_DIR / "hyundai_creta_2026.pdf")
    element_types = {e.element_type for e in doc.elements}
    assert "heading" in element_types
    assert "paragraph" in element_types
    assert "table" in element_types


def test_parse_pdf_table_rows_are_well_formed():
    _skip_if_no_brochures()
    doc = parse_pdf(BROCHURES_DIR / "hyundai_creta_2026.pdf")
    tables = [e for e in doc.elements if e.element_type == "table"]
    assert tables, "Expected at least one table to be extracted"
    for table in tables:
        assert table.table_rows is not None
        assert len(table.table_rows) >= 2  # header + at least one data row
        header_len = len(table.table_rows[0])
        assert all(len(row) == header_len for row in table.table_rows)


def test_parse_pdf_missing_file_returns_empty_document_without_raising():
    doc = parse_pdf(Path("/tmp/does_not_exist_drivewise.pdf"))
    assert doc.elements == []
    assert doc.car_brand == "Unknown Brand"


def test_table_text_is_not_duplicated_as_paragraph_text():
    """
    Regression test: a spec table's rows should appear exactly once, as a
    table element -- not also as free-floating paragraph text extracted by
    the text layer.
    """
    _skip_if_no_brochures()
    doc = parse_pdf(BROCHURES_DIR / "hyundai_creta_2026.pdf")
    tables = [e for e in doc.elements if e.element_type == "table"]
    paragraphs = [e for e in doc.elements if e.element_type == "paragraph"]

    for table in tables:
        first_row_joined = " | ".join(table.table_rows[1])  # a data row, not the header
        for paragraph in paragraphs:
            assert first_row_joined not in paragraph.text
