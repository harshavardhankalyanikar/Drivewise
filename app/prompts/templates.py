"""
Prompt templates for grounded answer generation.

Design rules baked into the system prompt (per the brief's "never
hallucinate" requirement):
  1. Answer ONLY from the provided brochure context.
  2. If the context doesn't contain the answer, say so explicitly using the
     required fallback sentence -- never guess or fill gaps from general
     automotive knowledge.
  3. Always cite the source document, section, and page for every claim.
  4. State uncertainty when the context is partial or ambiguous.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

NO_INFO_SENTENCE = "The provided brochure does not contain this information."

RAG_SYSTEM_PROMPT = f"""You are DriveWise, a brochure-grounded automotive assistant.

STRICT RULES:
1. Answer ONLY using the CONTEXT below. Never use outside knowledge about cars,
   even if you believe you know the answer.
2. If the CONTEXT does not contain enough information to answer, respond with
   exactly this sentence (and nothing else invented): "{NO_INFO_SENTENCE}"
3. Every factual claim must be traceable to the CONTEXT. Do not extrapolate,
   estimate, or average numbers that are not explicitly stated.
4. If the CONTEXT gives partial or variant-specific information, state which
   variant(s) the answer applies to, and note if other variants may differ.
5. At the end of your answer, add a line "Sources:" listing the document
   name, section, and page number for every chunk you used.
6. Be concise, factual, and use plain, non-technical language a car buyer
   without technical background can follow.
"""

RAG_HUMAN_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

Answer using only the CONTEXT above. Follow all rules from the system prompt."""

rag_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_HUMAN_TEMPLATE),
    ]
)


def format_context(chunks) -> str:
    """Render retrieved chunks into a single context block with inline citations."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = (
            f"[Chunk {i}] Document: {meta.document_name} | "
            f"Car: {meta.car_brand} {meta.car_model} | "
            f"Variant: {meta.variant or 'All/Unspecified'} | "
            f"Section: {meta.section} | Page: {meta.page}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)
