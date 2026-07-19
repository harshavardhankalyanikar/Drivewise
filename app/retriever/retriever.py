"""
Retrieval layer: metadata filtering + hybrid (semantic + BM25) search.

Order of operations for every query:
1. Metadata filter is applied first (brand/model/variant/fuel/section/page)
   -- this shrinks the candidate pool before any similarity math runs, which
   is both faster and more precise than filtering after the fact.
2. Semantic search (FAISS, dense embeddings) captures paraphrase / meaning
   matches ("how far can it go on a full tank" ~ "mileage").
3. BM25 (sparse, term-frequency) captures exact keyword/number matches
   ("SX(O)", "1497 cc") that dense embeddings can under-weight.
4. Scores are min-max normalised per method, then combined with configurable
   weights (`settings.hybrid_semantic_weight` / `hybrid_bm25_weight`) into a
   single hybrid score, giving the "EnsembleRetriever" behaviour referenced
   in the brief without depending on LangChain's specific implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config.schemas import Chunk, MetadataFilter, RetrievedChunk
from app.config.settings import settings
from app.utils.logger import get_logger
from app.vectorstore.faiss_store import FaissVectorStore

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().replace("/", " ").replace("-", " ").split()


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[Chunk]


class HybridRetriever:
    """Combines FAISS metadata-filtered semantic search with a BM25 keyword index."""

    def __init__(self, vector_store: FaissVectorStore, all_chunks: list[Chunk]):
        self.vector_store = vector_store
        self.all_chunks = all_chunks
        self._bm25_index = self._build_bm25_index(all_chunks)

    @staticmethod
    def _build_bm25_index(chunks: list[Chunk]) -> BM25Index:
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        return BM25Index(bm25=bm25, chunks=chunks)

    def _bm25_search(
        self, query: str, top_k: int, metadata_filter: MetadataFilter | None
    ) -> list[RetrievedChunk]:
        if self._bm25_index.bm25 is None:
            return []

        scores = self._bm25_index.bm25.get_scores(_tokenize(query))
        scored_chunks = list(zip(self._bm25_index.chunks, scores))
        scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

        results: list[RetrievedChunk] = []
        for chunk, score in scored_chunks:
            if metadata_filter and not FaissVectorStore._matches_filter(
                chunk.metadata.model_dump(), metadata_filter
            ):
                continue
            results.append(RetrievedChunk(text=chunk.text, metadata=chunk.metadata, bm25_score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def retrieve(
        self,
        query: str,
        metadata_filter: MetadataFilter | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k

        semantic_results = self.vector_store.similarity_search(query, top_k=top_k, metadata_filter=metadata_filter)
        bm25_results = self._bm25_search(query, top_k=top_k, metadata_filter=metadata_filter)

        # Merge by chunk_id, keeping both scores where available.
        merged: dict[str, RetrievedChunk] = {}
        for r in semantic_results:
            merged[r.metadata.chunk_id] = r
        for r in bm25_results:
            if r.metadata.chunk_id in merged:
                merged[r.metadata.chunk_id].bm25_score = r.bm25_score
            else:
                merged[r.metadata.chunk_id] = r

        chunk_ids = list(merged.keys())
        semantic_scores = [merged[cid].semantic_score or 0.0 for cid in chunk_ids]
        bm25_scores = [merged[cid].bm25_score or 0.0 for cid in chunk_ids]

        norm_semantic = _min_max_normalize(semantic_scores)
        norm_bm25 = _min_max_normalize(bm25_scores)

        for cid, ns, nb in zip(chunk_ids, norm_semantic, norm_bm25):
            hybrid = settings.hybrid_semantic_weight * ns + settings.hybrid_bm25_weight * nb
            merged[cid].hybrid_score = hybrid

        ranked = sorted(merged.values(), key=lambda r: r.hybrid_score or 0.0, reverse=True)

        if not ranked:
            logger.info(
                "No results for query=%r with filter=%s (either no match, or filters too narrow).",
                query, metadata_filter,
            )

        return ranked[:top_k]
