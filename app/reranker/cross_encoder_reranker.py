"""
Cross-encoder re-ranking.

Why re-rank at all: bi-encoders (used for the fast FAISS/BM25 first pass)
embed the query and each document *independently*, so similarity is a single
dot product between two fixed vectors -- cheap, but it can't model
fine-grained query-document interaction. A cross-encoder instead feeds the
(query, document) pair *together* through a transformer and outputs a single
relevance logit, letting it directly attend to how specific query terms
relate to specific document terms. This catches cases dense retrieval alone
often mis-ranks, e.g. distinguishing "SX(O) variant" from "SX variant" text
that are semantically very close in embedding space, or than the diesel spec
row is what actually answers a mileage question about the diesel variant.

The trade-off is cost: cross-encoders are far too slow to run over an entire
corpus, so the standard pattern -- used here -- is: retrieve top-20 cheaply,
re-rank only those with the cross-encoder, keep the top-5 for generation.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config.schemas import RetrievedChunk
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    logger.info("Loading cross-encoder re-ranker '%s'", model_name)
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.cross_encoder_model_name
        self._model = _load_cross_encoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        top_n = top_n or settings.rerank_top_n
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        reranked = sorted(candidates, key=lambda c: c.rerank_score or float("-inf"), reverse=True)
        return reranked[:top_n]
