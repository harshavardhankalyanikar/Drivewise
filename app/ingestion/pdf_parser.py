"""
PDF parsing layer.

Responsibilities:
- Open a brochure PDF with PyMuPDF (fast text + font-size based heading
  detection) and pdfplumber (reliable table extraction).
- Emit a flat, page-ordered list of `RawPageElement` objects: headings,
  paragraphs, and tables -- each tagged with its page number.
- Infer `car_brand` / `car_model` from the filename or first page, so
  downstream chunking can attach metadata without re-parsing.

This module never raises on a single bad file; it logs and returns an empty
element list so a batch ingestion job can continue with the remaining PDFs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from app.config.schemas import RawPageElement
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Known brands help us split "brand model" cleanly from filenames/titles.
KNOWN_BRANDS = [
    "Maruti Suzuki", "Maruti", "Hyundai", "Tata", "Mahindra", "Kia",
    "Toyota", "Honda", "Volkswagen", "Skoda", "MG", "Renault", "Nissan",
    "Ford", "Jeep", "Citroen",
]


@dataclass
class ParsedDocument:
    document_name: str
    car_brand: str
    car_model: str
    elements: list[RawPageElement]


def _infer_brand_model(document_name: str, first_page_text: str, full_text: str = "") -> tuple[str, str]:
    """Best-effort brand/model inference from filename, falling back to first page text, then full text."""
    stem = Path(document_name).stem.replace("_", " ").replace("-", " ")

    # 1. Search for brand in filename first
    for brand in KNOWN_BRANDS:
        pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)
        match = pattern.search(stem)
        if match:
            brand_norm = "Maruti Suzuki" if brand.lower() == "maruti" else brand
            rest = stem[match.end():].strip()
            rest = re.sub(r"\b(20\d{2}|brochure|official|product)\b", "", rest, flags=re.IGNORECASE)
            rest = re.sub(r"[^A-Za-z0-9\s]", " ", rest).strip()
            model = rest.split("\n")[0].strip()
            model = " ".join(model.split()[:3]) if model else "Unknown Model"
            return brand_norm, model or "Unknown Model"

    # 2. Search for brand in first page text
    for brand in KNOWN_BRANDS:
        pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)
        match = pattern.search(first_page_text[:200])
        if match:
            brand_norm = "Maruti Suzuki" if brand.lower() == "maruti" else brand
            clean_stem = re.sub(r"\b(20\d{2}|brochure|official|product)\b", "", stem, flags=re.IGNORECASE)
            clean_stem = re.sub(r"[^A-Za-z0-9\s]", " ", clean_stem).strip()
            model = " ".join(clean_stem.split()[:3]) if clean_stem else "Unknown Model"
            return brand_norm, model or "Unknown Model"

    # 3. Search for brand in the rest of the text
    for brand in KNOWN_BRANDS:
        pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)
        match = pattern.search(full_text)
        if match:
            brand_norm = "Maruti Suzuki" if brand.lower() == "maruti" else brand
            clean_stem = re.sub(r"\b(20\d{2}|brochure|official|product)\b", "", stem, flags=re.IGNORECASE)
            clean_stem = re.sub(r"[^A-Za-z0-9\s]", " ", clean_stem).strip()
            model = " ".join(clean_stem.split()[:3]) if clean_stem else "Unknown Model"
            return brand_norm, model or "Unknown Model"

    # Fallback if no brand is found anywhere: use stem as model
    clean_stem = re.sub(r"\b(20\d{2}|brochure|official|product)\b", "", stem, flags=re.IGNORECASE)
    clean_stem = re.sub(r"[^A-Za-z0-9\s]", " ", clean_stem).strip()
    model = " ".join(clean_stem.split()[:3]) if clean_stem else "Unknown Model"
    return "Unknown Brand", model


def _is_heading(span_size: float, body_size_estimate: float, text: str) -> bool:
    if len(text.strip()) == 0 or len(text) > 90:
        return False
    return span_size >= body_size_estimate + 1.5


def _line_in_any_bbox(line_bbox: list[float], table_bboxes: list[tuple[float, float, float, float]]) -> bool:
    """True if a text line's bbox center falls inside any known table bbox on the page."""
    if not table_bboxes or not line_bbox:
        return False
    cx = (line_bbox[0] + line_bbox[2]) / 2
    cy = (line_bbox[1] + line_bbox[3]) / 2
    for x0, y0, x1, y1 in table_bboxes:
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return True
    return False


def _extract_text_and_headings(
    pdf_path: Path, table_bboxes_by_page: dict[int, list[tuple[float, float, float, float]]]
) -> tuple[list[RawPageElement], str]:
    """
    Uses PyMuPDF span font sizes to distinguish headings from body paragraphs.

    `table_bboxes_by_page` (produced by `_extract_tables`) is used to skip any
    text line that physically falls inside a detected table region -- PyMuPDF
    has no notion of tables and would otherwise re-emit table cell text as
    ordinary paragraph text, duplicating every spec row as both a table chunk
    and a free-text chunk.
    """
    elements: list[RawPageElement] = []
    first_page_text = ""

    with fitz.open(pdf_path) as doc:
        # Estimate the "body" font size as the modal span size across the doc.
        sizes: list[float] = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["text"].strip():
                            sizes.append(round(span["size"], 1))
        body_size_estimate = max(set(sizes), key=sizes.count) if sizes else 10.0

        for page_index, page in enumerate(doc, start=1):
            page_lines_for_first_page: list[str] = []
            paragraph_buffer: list[str] = []
            paragraph_start_y: float | None = None
            page_table_bboxes = table_bboxes_by_page.get(page_index, [])

            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
                    if not line_text:
                        continue
                    line_bbox = line.get("bbox", [0, 0, 0, 0])
                    if _line_in_any_bbox(line_bbox, page_table_bboxes):
                        continue  # this text belongs to a table, already captured separately
                    max_span_size = max(
                        (span["size"] for span in line.get("spans", [])), default=body_size_estimate
                    )
                    line_y = line_bbox[1]

                    if page_index == 1:
                        page_lines_for_first_page.append(line_text)

                    if _is_heading(max_span_size, body_size_estimate, line_text):
                        # A heading closes out whatever paragraph text preceded it, so
                        # body text never bleeds across a section/subsection boundary.
                        if paragraph_buffer:
                            elements.append(
                                RawPageElement(
                                    page_number=page_index,
                                    element_type="paragraph",
                                    text=" ".join(paragraph_buffer),
                                    order_hint=paragraph_start_y or 0.0,
                                )
                            )
                            paragraph_buffer = []
                            paragraph_start_y = None
                        elements.append(
                            RawPageElement(
                                page_number=page_index,
                                element_type="heading",
                                text=line_text,
                                heading_level=1 if max_span_size >= body_size_estimate + 4 else 2,
                                order_hint=line_y,
                            )
                        )
                    else:
                        if paragraph_start_y is None:
                            paragraph_start_y = line_y
                        paragraph_buffer.append(line_text)

            if paragraph_buffer:
                elements.append(
                    RawPageElement(
                        page_number=page_index,
                        element_type="paragraph",
                        text=" ".join(paragraph_buffer),
                        order_hint=paragraph_start_y or 0.0,
                    )
                )

            if page_index == 1:
                first_page_text = "\n".join(page_lines_for_first_page)

    return elements, first_page_text


def _extract_tables(
    pdf_path: Path,
) -> tuple[list[RawPageElement], dict[int, list[tuple[float, float, float, float]]]]:
    """
    Extract tables via pdfplumber's `find_tables()` (not the separate
    `extract_tables()` call), so each table's extracted rows and its bounding
    box come from the *same* underlying Table object. Using two different
    detection calls can silently return tables in different order/counts,
    which would misalign row content with position -- and therefore with the
    surrounding section/heading during chunking.

    Returns both the table elements and a page -> [bbox, ...] map so the text
    extractor can exclude these regions and avoid duplicating table content.
    """
    table_elements: list[RawPageElement] = []
    bboxes_by_page: dict[int, list[tuple[float, float, float, float]]] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                found_tables = page.find_tables()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Table extraction failed on page %s of %s: %s", page_index, pdf_path.name, exc)
                continue

            for table in found_tables:
                try:
                    rows = table.extract()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed extracting one table on page %s of %s: %s", page_index, pdf_path.name, exc)
                    continue
                clean_rows = [
                    [cell.strip() if cell else "" for cell in row]
                    for row in rows
                    if any(cell and cell.strip() for cell in row)
                ]
                if len(clean_rows) < 2:
                    continue
                top_y = table.bbox[1]
                table_elements.append(
                    RawPageElement(
                        page_number=page_index,
                        element_type="table",
                        text="\n".join(" | ".join(row) for row in clean_rows),
                        table_rows=clean_rows,
                        order_hint=top_y,
                    )
                )
                bboxes_by_page.setdefault(page_index, []).append(tuple(table.bbox))

    return table_elements, bboxes_by_page


def parse_pdf(pdf_path: Path | str) -> ParsedDocument:
    """
    Parse a single brochure PDF into a page-ordered list of elements.

    Gracefully returns an empty-element ParsedDocument (rather than raising)
    if the file is missing, corrupted, or unreadable, so batch ingestion can
    continue with the remaining brochures.
    """
    pdf_path = Path(pdf_path)
    document_name = pdf_path.name

    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return ParsedDocument(document_name, "Unknown Brand", "Unknown Model", [])

    try:
        table_elements, table_bboxes_by_page = _extract_tables(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse tables from %s: %s", document_name, exc)
        table_elements, table_bboxes_by_page = [], {}

    try:
        text_elements, first_page_text = _extract_text_and_headings(pdf_path, table_bboxes_by_page)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse text/headings from %s: %s", document_name, exc)
        return ParsedDocument(document_name, "Unknown Brand", "Unknown Model", [])

    all_elements = sorted(text_elements + table_elements, key=lambda e: (e.page_number, e.order_hint))

    if not any(e.text.strip() for e in all_elements):
        logger.warning("No extractable content found in %s (possibly scanned/corrupted).", document_name)

    full_text = "\n".join(e.text for e in all_elements)
    brand, model = _infer_brand_model(document_name, first_page_text, full_text)
    logger.info("Parsed %s -> brand=%s model=%s elements=%d", document_name, brand, model, len(all_elements))

    return ParsedDocument(document_name, brand, model, all_elements)


def parse_pdf_directory(directory: Path | str) -> list[ParsedDocument]:
    """Parse every .pdf file in a directory, skipping and logging any failures."""
    directory = Path(directory)
    documents: list[ParsedDocument] = []
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)

    for pdf_file in pdf_files:
        documents.append(parse_pdf(pdf_file))

    return documents
