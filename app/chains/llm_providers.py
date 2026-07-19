"""
LLM generation provider.

`settings.llm_provider` selects the backend that answers questions:

- "groq"     -> Groq's chat completion API (requires GROQ_API_KEY). Groq runs
  open models (Llama 3.x, etc.) on their custom LPU hardware, so responses
  come back very fast -- a good fit for a chat-style RAG assistant.
- "template" -> a fully offline, deterministic extractive composer. This is
  the default so the whole pipeline (ingestion -> retrieval -> rerank ->
  "generation" -> sources) can be run, tested, and demoed with zero external
  API keys and zero network access -- and it still strictly obeys the
  never-hallucinate rule, since it only ever surfaces text that is already
  present in the retrieved chunks.

Both implement the same `generate(system_prompt, human_prompt) -> str`
signature so `chains/rag_chain.py` never needs to know which one is active.
"""

from __future__ import annotations

import re
from typing import Protocol

from app.config.schemas import RetrievedChunk
from app.config.settings import settings
from app.prompts.templates import NO_INFO_SENTENCE
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, human_prompt: str) -> str: ...


class GroqProvider:
    """Chat completion via Groq's OpenAI-compatible API."""

    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set; cannot use the 'groq' provider.")
        from groq import Groq  # local import: optional dependency

        self._client = Groq(api_key=settings.groq_api_key)

    def generate(self, system_prompt: str, human_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
        )
        return response.choices[0].message.content or ""


_STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "for", "on", "in", "to", "what", "does",
    "this", "car", "have", "has", "with", "and", "or", "which", "how", "much",
    "many", "it", "its", "do", "can", "i", "you", "we", "there",
}


def _keyword_overlap_score(query: str, text: str) -> int:
    query_terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _STOPWORDS}
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_terms & text_terms)


class TemplateAnswerComposer:
    """
    Deterministic, offline, extractive answer composer.

    It never invents a sentence: it only ever selects and lightly stitches
    together lines that already exist verbatim in the retrieved chunks
    (table rows and sentences), which makes hallucination structurally
    impossible. Used automatically if GROQ_API_KEY is not set.
    """

    def generate_from_chunks(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return NO_INFO_SENTENCE

        best_lines: list[tuple[int, str, RetrievedChunk]] = []
        for chunk in chunks:
            candidate_lines = (
                chunk.text.split("\n") if chunk.metadata.content_type == "table"
                else re.split(r"(?<=[.!?])\s+", chunk.text)
            )
            for line in candidate_lines:
                line = line.strip()
                if len(line) < 5:
                    continue
                score = _keyword_overlap_score(question, line)
                if score > 0:
                    best_lines.append((score, line, chunk))

        if not best_lines:
            return NO_INFO_SENTENCE

        best_lines.sort(key=lambda t: t[0], reverse=True)
        top_lines = best_lines[:4]

        answer_parts = []
        for _, line, chunk in top_lines:
            variant_note = f" (applies to: {chunk.metadata.variant})" if chunk.metadata.variant else ""
            answer_parts.append(f"- {line}{variant_note}")

        composed = "Based on the brochure:\n" + "\n".join(dict.fromkeys(answer_parts))
        return composed

    def generate(self, system_prompt: str, human_prompt: str) -> str:  # pragma: no cover - not used directly
        raise NotImplementedError("TemplateAnswerComposer uses generate_from_chunks(), not generate().")


def get_llm_provider() -> LLMProvider | TemplateAnswerComposer:
    provider = settings.llm_provider
    logger.info("Using LLM provider: %s", provider)
    if provider == "groq":
        return GroqProvider()
    return TemplateAnswerComposer()
