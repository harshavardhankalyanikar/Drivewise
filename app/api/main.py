"""
FastAPI application for DriveWise.

Endpoints:
  POST /upload    -- upload a new brochure PDF and re-index
  POST /query     -- ask a grounded question (full RAG: retrieve+rerank+generate)
  POST /search    -- raw retrieval only (no generation), for debugging/UI "view chunks"
  GET  /chunks    -- list/inspect chunks with optional metadata filters
  GET  /metadata  -- distinct brand/model/variant/section values, for building UI filters
  GET  /health    -- liveness/readiness probe
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.chains.pipeline import get_rag_chain, rebuild_index_from_brochures
from app.config.schemas import AnswerResponse, MetadataFilter, QueryRequest, RetrievedChunk
from app.config.settings import settings
from app.utils.logger import get_logger
from app.utils.persistence import load_chunks

logger = get_logger(__name__)

app = FastAPI(
    title="DriveWise API",
    description="Metadata-aware, brochure-grounded automotive RAG assistant.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    try:
        chain = get_rag_chain()
        ready = chain is not None
    except FileNotFoundError:
        ready = False
    return {"status": "ok", "index_ready": ready}


@app.post("/upload")
async def upload_brochure(file: UploadFile = File(...)) -> dict:
    """Save an uploaded brochure PDF and rebuild the FAISS index from all brochures."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    destination = Path(settings.brochures_dir) / file.filename
    try:
        with open(destination, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save uploaded file %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc
    finally:
        file.file.close()

    try:
        chunk_count = rebuild_index_from_brochures()
    except Exception as exc:  # noqa: BLE001
        logger.error("Re-indexing failed after upload of %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {exc}") from exc

    return {
        "filename": file.filename,
        "saved_to": str(destination),
        "reindexed": True,
        "total_chunks": chunk_count,
    }


@app.post("/query", response_model=AnswerResponse)
def query(request: QueryRequest) -> AnswerResponse:
    """Full grounded RAG pipeline: filter -> hybrid retrieve -> rerank -> generate -> cite."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be empty.")

    try:
        chain = get_rag_chain()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Run `python scripts/ingest.py` or POST /upload first.",
        ) from exc

    try:
        return chain.answer(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed for question=%r", request.question)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.post("/search", response_model=list[RetrievedChunk])
def search(request: QueryRequest) -> list[RetrievedChunk]:
    """Retrieval-only endpoint (hybrid + metadata filter, no LLM generation)."""
    try:
        chain = get_rag_chain()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Index not built yet.") from exc

    filters = MetadataFilter(
        car_brand=request.car_brand,
        car_model=request.car_model,
        variant=request.variant,
        fuel_type=request.fuel_type,
        section=request.section,
    )
    top_k = request.top_k or settings.retrieval_top_k
    return chain.retriever.retrieve(request.question, metadata_filter=filters, top_k=top_k)


@app.get("/chunks")
def list_chunks(
    car_brand: str | None = Query(default=None),
    car_model: str | None = Query(default=None),
    section: str | None = Query(default=None),
    limit: int = Query(default=50, le=500),
) -> dict:
    try:
        chunks = load_chunks()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="No chunks found. Run ingestion first.") from exc

    def _matches(c) -> bool:
        if car_brand and c.metadata.car_brand.lower() != car_brand.lower():
            return False
        if car_model and c.metadata.car_model.lower() != car_model.lower():
            return False
        if section and c.metadata.section.lower() != section.lower():
            return False
        return True

    filtered = [c for c in chunks if _matches(c)][:limit]
    return {"count": len(filtered), "chunks": [c.model_dump() for c in filtered]}


@app.get("/metadata")
def metadata_facets() -> dict:
    """Distinct brand/model/variant/fuel/section values -- used to populate UI filter dropdowns."""
    try:
        chunks = load_chunks()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="No chunks found. Run ingestion first.") from exc

    brands = sorted({c.metadata.car_brand for c in chunks})
    models = sorted({c.metadata.car_model for c in chunks})
    variants = sorted({c.metadata.variant for c in chunks if c.metadata.variant})
    fuel_types = sorted({c.metadata.fuel_type for c in chunks if c.metadata.fuel_type})
    sections = sorted({c.metadata.section for c in chunks})
    documents = sorted({c.metadata.document_name for c in chunks})

    brand_model_map: dict[str, list[str]] = {}
    for c in chunks:
        brand_model_map.setdefault(c.metadata.car_brand, set()).add(c.metadata.car_model)
    brand_model_map = {k: sorted(v) for k, v in brand_model_map.items()}

    return {
        "car_brands": brands,
        "car_models": models,
        "variants": variants,
        "fuel_types": fuel_types,
        "sections": sections,
        "documents": documents,
        "brand_to_models": brand_model_map,
    }
