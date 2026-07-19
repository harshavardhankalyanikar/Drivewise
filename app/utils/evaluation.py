"""
Evaluation utilities for DriveWise RAG pipeline.

Contains calculations for:
1. RAGAS Metrics (using Groq)
2. Retrieval Metrics (Precision@K, Recall@K, MRR, Hit Rate@K)
3. DeepEval Metrics (using Groq)
4. Source Attribution Validation
5. Timing & Latency tracking
6. CSV Logging
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from deepeval.metrics import AnswerRelevancyMetric as DEAnswerRelevancyMetric
from deepeval.metrics import FaithfulnessMetric as DEFaithfulnessMetric
from deepeval.metrics import HallucinationMetric as DEHallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel
import sys
from types import ModuleType

# Mock missing VertexAI langchain imports to prevent Ragas from failing on startup
for name in ["langchain_community.chat_models.vertexai", "langchain_community.embeddings.vertexai"]:
    if name not in sys.modules:
        m = ModuleType(name)
        m.ChatVertexAI = type("ChatVertexAI", (object,), {})
        m.VertexAIEmbeddings = type("VertexAIEmbeddings", (object,), {})
        sys.modules[name] = m

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall

from app.config.schemas import AnswerResponse, RetrievedChunk
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# =====================================================================
# DeepEval Groq Adapter
# =====================================================================

class DeepEvalGroq(DeepEvalBaseLLM):
    """DeepEval custom LLM adapter to route evaluation requests to Groq."""

    def __init__(self, chat_model: ChatGroq) -> None:
        self.chat_model = chat_model

    def load_model(self) -> ChatGroq:
        return self.chat_model

    def generate(self, prompt: str) -> str:
        try:
            return self.chat_model.invoke(prompt).content
        except Exception as e:
            logger.error("DeepEval Groq generation failed: %s", e)
            return ""

    async def a_generate(self, prompt: str) -> str:
        try:
            res = await self.chat_model.ainvoke(prompt)
            return res.content
        except Exception as e:
            logger.error("DeepEval Groq async generation failed: %s", e)
            return ""

    def get_model_name(self) -> str:
        return f"Groq-{settings.groq_model}"


# =====================================================================
# Retrieval Metrics
# =====================================================================

def calculate_retrieval_metrics(
    retrieved: list[RetrievedChunk], expected_sources: list[dict[str, Any]], k: int = 5
) -> dict[str, float]:
    if not expected_sources:
        return {f"precision_{k}": 0.0, f"recall_{k}": 0.0, "mrr": 0.0, "hit_rate": 0.0}

    top_retrieved = retrieved[:k]

    def _is_match(ret, exp):
        doc_match = ret.metadata.document_name.strip().lower() == exp.get("document_name", "").strip().lower()
        page_match = int(ret.metadata.page) == int(exp.get("page", 0))
        chunk_match = ret.metadata.chunk_id == exp.get("chunk_id")
        return chunk_match or (doc_match and page_match)

    # Precision@k: judge each retrieved chunk independently -- fine as-is,
    # since the denominator is k, not len(expected_sources).
    relevant_retrieved_count = 0
    first_hit_rank = 0
    for idx, ret in enumerate(top_retrieved):
        if any(_is_match(ret, exp) for exp in expected_sources):
            relevant_retrieved_count += 1
            if first_hit_rank == 0:
                first_hit_rank = idx + 1

    # Recall@k: count DISTINCT expected sources found at least once in top-k.
    # Must dedupe -- otherwise a page split into a table chunk + paragraph
    # chunk (both matching the same single expected entry) inflates recall
    # past 1.0, exactly like the 1.85 you saw.
    unique_expected_found = sum(
        1 for exp in expected_sources if any(_is_match(ret, exp) for ret in top_retrieved)
    )

    precision = relevant_retrieved_count / k
    recall = unique_expected_found / len(expected_sources)
    hit_rate = 1.0 if relevant_retrieved_count > 0 else 0.0
    mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

    return {
        f"precision_{k}": round(precision, 4),
        f"recall_{k}": round(min(recall, 1.0), 4),  # defensive clamp only; shouldn't trigger now
        "mrr": round(mrr, 4),
        "hit_rate": round(hit_rate, 4),
    }


# =====================================================================
# Source Attribution Validation
# =====================================================================

def validate_source_attribution(
    document_name: str, page_number: int, snippet: str, generated_answer: str
) -> dict[str, Any]:
    """
    Verify source file existence, page validity, and chunk keyword alignment.
    Generates a confidence score between 0.0 and 1.0.
    """
    file_exists = False
    page_valid = False
    content_match = False
    confidence = 1.0
    reason = []

    # 1. Check if file exists on disk
    doc_path = Path(settings.brochures_dir) / document_name
    if doc_path.exists():
        file_exists = True
    else:
        confidence -= 0.4
        reason.append("Source PDF file not found on disk.")

    # 2. Check if page number is valid in PDF
    if file_exists:
        try:
            with fitz.open(doc_path) as doc:
                if 1 <= page_number <= len(doc):
                    page_valid = True
                else:
                    confidence -= 0.3
                    reason.append(f"Page number {page_number} exceeds PDF pages ({len(doc)}).")
        except Exception as e:
            confidence -= 0.3
            reason.append(f"Failed to open PDF to verify page: {e}")
    else:
        confidence -= 0.3
        reason.append("Cannot verify page validity because PDF file is missing.")

    # 3. Check if retrieved chunk actually contains/supports the answer (keyword overlap heuristic)
    answer_words = {w.lower() for w in generated_answer.split() if len(w) > 3}
    snippet_words = {w.lower() for w in snippet.split() if len(w) > 3}
    
    overlap = len(answer_words & snippet_words)
    if len(answer_words) > 0 and (overlap / len(answer_words)) >= 0.15:
        content_match = True
    elif "no information" in generated_answer.lower() or "not mention" in generated_answer.lower():
        content_match = True  # Extractive composer template returned no info, so it's correct
    else:
        confidence -= 0.3
        reason.append("Low keyword correlation between answer and source snippet.")

    confidence = round(max(0.0, confidence), 2)
    return {
        "file_exists": file_exists,
        "page_valid": page_valid,
        "content_match": content_match,
        "confidence_score": confidence,
        "reason": "; ".join(reason) if reason else "Valid source attribution.",
    }


# =====================================================================
# RAGAS and DeepEval Evaluators
# =====================================================================

class DriveWiseEvaluator:
    """Manages LLM-based evaluation metrics using RAGAS and DeepEval via Groq."""

    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required to initialize evaluations.")

        # Initialize ChatGroq LLM
        self.llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=0.0
        )
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            encode_kwargs={"device": settings.embedding_device}
        )

        # Configure RAGAS metrics to use Groq LLM and Embeddings
        faithfulness.llm = self.llm
        answer_relevancy.llm = self.llm
        answer_relevancy.embeddings = self.embeddings
        context_precision.llm = self.llm
        context_precision.embeddings = self.embeddings
        context_recall.llm = self.llm
        context_recall.embeddings = self.embeddings

        # Configure DeepEval to use Groq LLM
        self.deepeval_llm = DeepEvalGroq(self.llm)
        self.de_faithfulness = DEFaithfulnessMetric(threshold=0.5, model=self.deepeval_llm)
        self.de_relevancy = DEAnswerRelevancyMetric(threshold=0.5, model=self.deepeval_llm)
        self.de_hallucination = DEHallucinationMetric(threshold=0.5, model=self.deepeval_llm)

    def evaluate_ragas(
        self, question: str, answer: str, contexts: list[str], ground_truth: str
    ) -> dict[str, float]:
        """Runs RAGAS metrics on a single Q&A turn."""
        from datasets import Dataset

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)
        
        try:
            results = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            return {
                "faithfulness": round(float(results.get("faithfulness", 0.0)), 4),
                "answer_relevancy": round(float(results.get("answer_relevancy", 0.0)), 4),
                "context_precision": round(float(results.get("context_precision", 0.0)), 4),
                "context_recall": round(float(results.get("context_recall", 0.0)), 4),
            }
        except Exception as e:
            logger.error("Ragas evaluation failed: %s", e)
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }

    def evaluate_deepeval(
        self, question: str, answer: str, contexts: list[str]
    ) -> dict[str, float]:
        """Runs DeepEval metrics on a single Q&A turn."""
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts
        )

        try:
            # Measure Faithfulness
            self.de_faithfulness.measure(test_case)
            f_score = self.de_faithfulness.score
            
            # Measure Relevancy
            self.de_relevancy.measure(test_case)
            r_score = self.de_relevancy.score
            
            # Measure Hallucination
            self.de_hallucination.measure(test_case)
            h_score = self.de_hallucination.score

            return {
                "deepeval_faithfulness": round(float(f_score), 4),
                "deepeval_relevancy": round(float(r_score), 4),
                "deepeval_hallucination": round(float(h_score), 4),
            }
        except Exception as e:
            logger.error("DeepEval evaluation failed: %s", e)
            return {
                "deepeval_faithfulness": 0.0,
                "deepeval_relevancy": 0.0,
                "deepeval_hallucination": 0.0,
            }


# =====================================================================
# Logging Results
# =====================================================================

def log_evaluation_result(row: dict[str, Any]) -> None:
    """Appends a row of evaluation results to evaluation_results.csv."""
    path = Path(settings.evaluation_results_path)
    file_exists = path.exists()

    headers = [
        "Question", "Expected Answer", "Generated Answer", "Retrieved Context", 
        "Retrieved Sources", "Faithfulness", "Answer Relevancy", "Context Precision", 
        "Context Recall", "Precision@5", "Recall@5", "MRR", "Hit Rate", 
        "Hallucination Score", "Retrieval Time", "Reranking Time", 
        "LLM Generation Time", "Latency", "Timestamp"
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "Question": row.get("question", ""),
            "Expected Answer": row.get("expected_answer", ""),
            "Generated Answer": row.get("generated_answer", ""),
            "Retrieved Context": row.get("retrieved_context", ""),
            "Retrieved Sources": row.get("retrieved_sources", ""),
            "Faithfulness": row.get("faithfulness", 0.0),
            "Answer Relevancy": row.get("answer_relevancy", 0.0),
            "Context Precision": row.get("context_precision", 0.0),
            "Context Recall": row.get("context_recall", 0.0),
            "Precision@5": row.get("precision_5", 0.0),
            "Recall@5": row.get("recall_5", 0.0),
            "MRR": row.get("mrr", 0.0),
            "Hit Rate": row.get("hit_rate", 0.0),
            "Hallucination Score": row.get("hallucination_score", 0.0),
            "Retrieval Time": row.get("retrieval_time", 0.0),
            "Reranking Time": row.get("reranking_time", 0.0),
            "LLM Generation Time": row.get("llm_generation_time", 0.0),
            "Latency": row.get("latency", 0.0),
            "Timestamp": row.get("timestamp", ""),
        })
