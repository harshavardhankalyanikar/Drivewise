"""Persist/load the chunk list as JSONL, independent of the FAISS index files."""

from __future__ import annotations

import json
from pathlib import Path

from app.config.schemas import Chunk
from app.config.settings import settings


def save_chunks(chunks: list[Chunk], path: Path | str | None = None) -> Path:
    path = Path(path or (settings.processed_dir / "chunks.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    return path


def load_chunks(path: Path | str | None = None) -> list[Chunk]:
    path = Path(path or (settings.processed_dir / "chunks.jsonl"))
    if not path.exists():
        raise FileNotFoundError(f"No persisted chunks found at {path}. Run the ingestion script first.")
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks
