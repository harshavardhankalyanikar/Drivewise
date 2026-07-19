"""
Intelligent, metadata-aware chunking.

Rather than fixed-length sliding-window chunking, this module chunks
*structurally*:

- Headings become section/subsection boundaries (heading-aware).
- Consecutive body text under a heading is grouped into paragraph-aware
  chunks, soft-capped at `settings.max_chunk_chars` with a small character
  overlap so a fact split across the cap boundary isn't lost.
- Tables are kept intact as their own chunk (table-aware) -- splitting a
  specs table would destroy the row/column relationship that makes it
  answerable ("What is the mileage?").
- Every chunk is stamped with `car_brand`, `car_model`, `variant`,
  `fuel_type`, `page`, `section`, `heading`, `document_name`, `chunk_id`
  (metadata-aware / specifications-aware chunking).
"""

from __future__ import annotations

import hashlib
import re

from app.config.schemas import Chunk, ChunkMetadata, RawPageElement, SectionType
from app.config.settings import settings
from app.ingestion.pdf_parser import ParsedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SECTION_KEYWORDS: dict[SectionType, list[str]] = {
    SectionType.ENGINE_PERFORMANCE: ["engine", "performance", "power", "torque", "transmission", "drivetrain"],
    SectionType.MILEAGE_FUEL: ["mileage", "fuel efficiency", "arai", "fuel tank", "km/l", "km/kg"],
    SectionType.SAFETY: ["safety", "airbag", "adas", "ncap", "abs", "esc", "crash"],
    SectionType.DIMENSIONS: ["dimension", "length", "width", "height", "wheelbase", "ground clearance", "boot space"],
    SectionType.INTERIOR_COMFORT: ["interior", "comfort", "sunroof", "seat", "climate control", "upholstery"],
    SectionType.INFOTAINMENT: ["infotainment", "connectivity", "touchscreen", "android auto", "apple carplay", "sound system"],
}

_FUEL_KEYWORDS = {
    "petrol": ["petrol", "mpi"],
    "turbo_petrol": ["turbo-petrol", "turbo petrol"],
    "diesel": ["diesel", "crdi"],
    "cng": ["cng"],
    "electric": ["electric", "ev "],
}

_VARIANT_LIST_PATTERN = re.compile(r"available variants\s*:\s*(.+)", re.IGNORECASE)


def _classify_section(heading_text: str | None, body_text: str) -> SectionType:
    haystack = f"{heading_text or ''} {body_text}".lower()
    best_section = SectionType.GENERAL
    best_hits = 0
    for section, keywords in _SECTION_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in haystack)
        if hits > best_hits:
            best_hits = hits
            best_section = section
    return best_section


def _detect_fuel_type(text: str) -> str | None:
    lower = text.lower()
    matches = [ft for ft, kws in _FUEL_KEYWORDS.items() if any(kw in lower for kw in kws)]
    # If turbo_petrol matched, prefer it over the generic petrol match.
    if "turbo_petrol" in matches:
        return "turbo_petrol"
    if len(matches) == 1:
        return matches[0]
    return None


def _extract_variant_list(elements: list[RawPageElement]) -> list[str]:
    for element in elements:
        if element.page_number != 1:
            continue
        match = _VARIANT_LIST_PATTERN.search(element.text)
        if match:
            raw = match.group(1)
            variants = [v.strip() for v in raw.split(",") if v.strip()]
            return variants
    return []


def _detect_variant(text: str, known_variants: list[str]) -> str | None:
    if not known_variants:
        return None
    # Sort longer names first so "SX(O)" isn't shadowed by a substring match of "SX".
    sorted_variants = sorted(known_variants, key=len, reverse=True)
    found = []
    remaining = text
    for variant in sorted_variants:
        # Custom boundary check (not \b) since variant codes can end in
        # punctuation like "SX(O)", where \b's word/non-word transition
        # rule would incorrectly fail to match before a following space.
        # Requiring no adjacent letter/digit prevents single-letter codes
        # like "E" from matching inside ordinary words (e.g. "Available").
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(variant) + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        match = pattern.search(remaining)
        if match:
            found.append(variant)
            remaining = pattern.sub(" ", remaining, count=1)
    unique_found = list(dict.fromkeys(found))
    if len(unique_found) == 1:
        return unique_found[0]
    return None  # applies to all variants, or ambiguous -> leave unscoped


def _make_chunk_id(document_name: str, page: int, index: int) -> str:
    raw = f"{document_name}:{page}:{index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split overly long paragraph text on sentence boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # start next chunk with a small overlap tail from the previous chunk
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_document(parsed_document: ParsedDocument) -> list[Chunk]:
    """Convert a ParsedDocument's flat element list into metadata-rich Chunks."""
    chunks: list[Chunk] = []
    known_variants = _extract_variant_list(parsed_document.elements)

    current_section_heading: str | None = None
    current_subheading: str | None = None
    chunk_index = 0

    for element in parsed_document.elements:
        if element.element_type == "heading":
            if element.heading_level == 1:
                current_section_heading = element.text
                current_subheading = None
            else:
                current_subheading = element.text
            continue

        section_type = _classify_section(current_section_heading, element.text)
        heading_for_chunk = current_subheading or current_section_heading

        if element.element_type == "table":
            variant = _detect_variant(element.text, known_variants)
            fuel_type = _detect_fuel_type(element.text)
            chunk_index += 1
            chunks.append(
                Chunk(
                    text=element.text,
                    metadata=ChunkMetadata(
                        chunk_id=_make_chunk_id(parsed_document.document_name, element.page_number, chunk_index),
                        document_name=parsed_document.document_name,
                        car_brand=parsed_document.car_brand,
                        car_model=parsed_document.car_model,
                        variant=variant,
                        fuel_type=fuel_type,
                        page=element.page_number,
                        section=section_type.value,
                        heading=heading_for_chunk,
                        content_type="table",
                    ),
                )
            )
            continue

        # paragraph: apply length-aware splitting, each split keeps identical metadata
        if len(element.text.strip()) < settings.min_chunk_chars:
            continue

        pieces = _split_long_text(element.text, settings.max_chunk_chars, settings.chunk_overlap_chars)
        for piece in pieces:
            variant = _detect_variant(piece, known_variants)
            fuel_type = _detect_fuel_type(piece)
            chunk_index += 1
            chunks.append(
                Chunk(
                    text=piece,
                    metadata=ChunkMetadata(
                        chunk_id=_make_chunk_id(parsed_document.document_name, element.page_number, chunk_index),
                        document_name=parsed_document.document_name,
                        car_brand=parsed_document.car_brand,
                        car_model=parsed_document.car_model,
                        variant=variant,
                        fuel_type=fuel_type,
                        page=element.page_number,
                        section=section_type.value,
                        heading=heading_for_chunk,
                        content_type="text",
                    ),
                )
            )

    logger.info("Chunked %s -> %d chunks", parsed_document.document_name, len(chunks))
    return chunks


def chunk_documents(parsed_documents: list[ParsedDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in parsed_documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
