"""
Centralised, typed application configuration.

All configurable knobs (embedding model, chunk sizes, LLM provider, retrieval
depth, etc.) live here and are overridable via environment variables or a
`.env` file, so nothing is hard-coded deep inside the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide settings, sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -----------------------------------------------------------
    base_dir: Path = BASE_DIR
    brochures_dir: Path = BASE_DIR / "data" / "brochures"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    faiss_index_dir: Path = BASE_DIR / "data" / "processed" / "faiss_index"
    logs_dir: Path = BASE_DIR / "logs"
    evaluation_results_path: Path = BASE_DIR / "evaluation_results.csv"
    evaluation_dataset_path: Path = BASE_DIR / "data" / "evaluation_dataset.json"

    # --- Embeddings --------------------------------------------------------
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model id. Swap to BAAI/bge-small-en-v1.5 etc.",
    )
    embedding_device: str = Field(default="cpu")

    # --- Chunking ----------------------------------------------------------
    max_chunk_chars: int = Field(default=1200, description="Soft cap for a single chunk")
    chunk_overlap_chars: int = Field(default=150)
    min_chunk_chars: int = Field(default=40, description="Below this, a chunk gets merged forward")

    # --- Retrieval -----------------------------------------------------------
    retrieval_top_k: int = Field(default=20, description="Candidates pulled before re-ranking")
    rerank_top_n: int = Field(default=5, description="Chunks kept after cross-encoder re-ranking")
    hybrid_semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    hybrid_bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)

    # --- Re-ranker -----------------------------------------------------------
    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # --- LLM / generation ------------------------------------------------------
    llm_provider: Literal["groq", "template"] = Field(
        default="template",
        description=(
            "'template' runs a fully offline, deterministic extractive answer "
            "composer (no API key needed) -- used as a safe default and test "
            "fallback. Set to 'groq' to use Groq's hosted LLM API."
        ),
    )
    groq_api_key: str | None = Field(default=None)
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    llm_temperature: float = Field(default=0.1)
    openai_api_key: str | None = Field(default=None)

    # --- API / App -------------------------------------------------------------
    app_name: str = Field(default="DriveWise")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Logging -----------------------------------------------------------
    log_level: str = Field(default="INFO")


settings = Settings()

for _dir in (
    settings.brochures_dir,
    settings.processed_dir,
    settings.uploads_dir,
    settings.faiss_index_dir,
    settings.logs_dir,
):
    os.makedirs(_dir, exist_ok=True)
