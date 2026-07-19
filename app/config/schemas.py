"""
Shared Pydantic data models.

Keeping these in one module (instead of redefining dicts everywhere) gives us
validation, IDE autocomplete, and a single source of truth for what a "chunk"
or an "answer" looks like as it flows through parser -> chunker -> embedder ->
vector store -> retriever -> reranker -> chain -> API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SectionType(str, Enum):
    ENGINE_PERFORMANCE = "engine_and_performance"
    MILEAGE_FUEL = "mileage_and_fuel_efficiency"
    SAFETY = "safety_features"
    DIMENSIONS = "dimensions"
    INTERIOR_COMFORT = "interior_and_comfort"
    INFOTAINMENT = "infotainment_and_connectivity"
    GENERAL = "general"


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk, used for filtering and citation."""

    chunk_id: str
    document_name: str
    car_brand: str
    car_model: str
    variant: str | None = None  # None => applies to all/multiple variants
    fuel_type: str | None = None
    transmission: str | None = None
    page: int
    section: str
    heading: str | None = None
    content_type: str = Field(default="text", description="text | table | heading")


class Chunk(BaseModel):
    """A single retrievable unit of brochure content."""

    text: str
    metadata: ChunkMetadata


class RawPageElement(BaseModel):
    """An intermediate representation emitted by the PDF parser, pre-chunking."""

    page_number: int
    element_type: str  # "heading" | "paragraph" | "table"
    text: str
    heading_level: int | None = None
    table_rows: list[list[str]] | None = None
    order_hint: float = Field(
        default=0.0,
        description="Vertical (top-to-bottom) position on the page, used to "
        "interleave text and table elements in true reading order.",
    )


class MetadataFilter(BaseModel):
    """User/API-supplied filter constraints applied before semantic search."""

    car_brand: str | None = None
    car_model: str | None = None
    variant: str | None = None
    fuel_type: str | None = None
    section: str | None = None
    page: int | None = None


class RetrievedChunk(BaseModel):
    text: str
    metadata: ChunkMetadata
    semantic_score: float | None = None
    bm25_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None


class SourceCitation(BaseModel):
    document_name: str
    car_brand: str
    car_model: str
    section: str
    page: int
    heading: str | None = None
    chunk_id: str
    snippet: str


class QueryRequest(BaseModel):
    question: str
    car_brand: str | None = None
    car_model: str | None = None
    variant: str | None = None
    fuel_type: str | None = None
    section: str | None = None
    top_k: int | None = None


class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[SourceCitation]
    reasoning_summary: str
    retrieved_chunk_count: int
    metadata_filters_applied: dict[str, Any]
    grounded: bool
