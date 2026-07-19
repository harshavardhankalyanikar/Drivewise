"""
End-to-end RAG chain.

Wires together every layer built so far:

    MetadataFilter -> HybridRetriever (FAISS + BM25) -> CrossEncoderReranker
        -> prompt formatting -> LLMProvider -> AnswerResponse (with sources,
        confidence, and a reasoning summary)

Built with LangChain's LCEL (`RunnableLambda` / `|`) for the LLM-backed path,
so swapping providers or inserting extra steps (e.g. a query-rewriting step)
is a matter of composing another Runnable into the pipeline.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from app.chains.llm_providers import TemplateAnswerComposer, get_llm_provider
from app.config.schemas import (
    AnswerResponse,
    MetadataFilter,
    QueryRequest,
    RetrievedChunk,
    SourceCitation,
)
from app.config.settings import settings
from app.prompts.templates import NO_INFO_SENTENCE, format_context, rag_chat_prompt
from app.reranker.cross_encoder_reranker import CrossEncoderReranker
from app.retriever.retriever import HybridRetriever
from app.utils.logger import get_logger, query_monitor

logger = get_logger(__name__)


def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
    sources = []
    for chunk in chunks:
        meta = chunk.metadata
        snippet = chunk.text.strip().replace("\n", " ")
        snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")
        sources.append(
            SourceCitation(
                document_name=meta.document_name,
                car_brand=meta.car_brand,
                car_model=meta.car_model,
                section=meta.section,
                page=meta.page,
                heading=meta.heading,
                chunk_id=meta.chunk_id,
                snippet=snippet,
            )
        )
    return sources


def _confidence_from_scores(chunks: list[RetrievedChunk]) -> float:
    """
    Confidence heuristic: normalised average of the top re-ranked chunks'
    cross-encoder scores, squashed into [0, 1] with a logistic-style clip.
    This is a proxy signal for the UI, not a calibrated probability.
    """
    if not chunks:
        return 0.0
    scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if not scores:
        return 0.4  # some retrieval happened but no rerank score available
    avg = sum(scores) / len(scores)
    # ms-marco cross-encoder logits are roughly in [-11, 11]; squash to [0,1]
    squashed = 1 / (1 + pow(2.718281828, -avg))
    return round(min(max(squashed, 0.05), 0.99), 3)


def _reasoning_summary(question: str, filters: MetadataFilter, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"No brochure chunks matched the query after metadata filtering "
            f"({filters.model_dump(exclude_none=True)}) and hybrid retrieval."
        )
    sections = sorted({c.metadata.section for c in chunks})
    docs = sorted({c.metadata.document_name for c in chunks})
    return (
        f"Retrieved {len(chunks)} chunk(s) from {len(docs)} document(s) "
        f"covering section(s) {', '.join(sections)}; re-ranked by cross-encoder "
        f"relevance to: '{question}'."
    )


class RAGChain:
    def __init__(self, retriever: HybridRetriever, reranker: CrossEncoderReranker | None = None):
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_provider = get_llm_provider()

        if not isinstance(self.llm_provider, TemplateAnswerComposer):
            self._llm_runnable = (
                rag_chat_prompt
                | RunnableLambda(
                    lambda prompt_value: self.llm_provider.generate(
                        system_prompt=prompt_value.messages[0].content,
                        human_prompt=prompt_value.messages[1].content,
                    )
                )
                | StrOutputParser()
            )
        else:
            self._llm_runnable = None

    def answer(self, request: QueryRequest) -> AnswerResponse:
        filters = MetadataFilter(
            car_brand=request.car_brand,
            car_model=request.car_model,
            variant=request.variant,
            fuel_type=request.fuel_type,
            section=request.section,
        )

        with query_monitor.track(request.question, filters.model_dump(exclude_none=True)) as record:
            top_k = request.top_k or settings.retrieval_top_k
            candidates = self.retriever.retrieve(request.question, metadata_filter=filters, top_k=top_k)
            reranked = self.reranker.rerank(request.question, candidates, top_n=settings.rerank_top_n)

            record["retrieved_count"] = len(candidates)
            record["reranked_count"] = len(reranked)

            if isinstance(self.llm_provider, TemplateAnswerComposer):
                raw_answer = self.llm_provider.generate_from_chunks(request.question, reranked)
            else:
                context = format_context(reranked)
                raw_answer = self._llm_runnable.invoke({"context": context, "question": request.question})

            grounded = NO_INFO_SENTENCE not in raw_answer and len(reranked) > 0
            confidence = _confidence_from_scores(reranked) if grounded else 0.0

            record["grounded"] = grounded
            record["confidence"] = confidence

            response = AnswerResponse(
                answer=raw_answer,
                confidence=confidence,
                sources=_build_sources(reranked) if grounded else [],
                reasoning_summary=_reasoning_summary(request.question, filters, reranked),
                retrieved_chunk_count=len(reranked),
                metadata_filters_applied=filters.model_dump(exclude_none=True),
                grounded=grounded,
            )
            return response
