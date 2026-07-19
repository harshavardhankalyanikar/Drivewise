"""
Structured logging + lightweight query/monitoring log.

Two things live here:
1. `get_logger` -- a standard structured logger (console + rotating file).
2. `QueryMonitor` -- appends a JSON line per query with latency, retrieval
   counts, and failure status, satisfying the "logging & monitoring" and
   "evaluation tracking" requirements from the brief without needing an
   external observability stack.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

from app.config.settings import settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        root = logging.getLogger()
        root.setLevel(settings.log_level)

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

        file_handler = RotatingFileHandler(
            Path(settings.logs_dir) / "drivewise.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        _CONFIGURED = True

    return logger


class QueryMonitor:
    """Appends JSONL records for every user query: latency, retrieval status, failures."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or (Path(settings.logs_dir) / "query_log.jsonl")

    def _write(self, record: dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @contextmanager
    def track(self, question: str, filters: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """
        Usage:
            with monitor.track(question, filters) as record:
                ... do retrieval + generation, mutate `record` with results ...
        Automatically records latency and whether an exception occurred.
        """
        record: dict[str, Any] = {
            "question": question,
            "filters": filters,
            "status": "success",
        }
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:  # noqa: BLE001 - deliberately broad for logging
            record["status"] = "failed"
            record["error"] = str(exc)
            raise
        finally:
            record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            record["timestamp"] = time.time()
            self._write(record)


query_monitor = QueryMonitor()
