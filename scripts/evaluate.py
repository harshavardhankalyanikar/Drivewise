"""
CLI script to execute the evaluation runner on the ground truth dataset.

Run with:
    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Adjust path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chains.pipeline import get_rag_chain
from app.config.settings import settings
from app.prompts.templates import format_context
from app.utils.evaluation import (
    DriveWiseEvaluator,
    calculate_retrieval_metrics,
    log_evaluation_result,
    validate_source_attribution,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation on the ground truth dataset.")
    parser.add_argument("--dataset", type=str, default=str(settings.evaluation_dataset_path))
    parser.add_argument("--output", type=str, default=str(settings.evaluation_results_path))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found at %s. Please create it first.", dataset_path)
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info("Loaded %d test cases from %s", len(dataset), dataset_path)

    # Initialize chain and evaluator
    try:
        chain = get_rag_chain()
    except FileNotFoundError:
        logger.error("FAISS index not built yet. Please run ingest script first: python scripts/ingest.py")
        sys.exit(1)

    logger.info("Initializing Ragas & DeepEval evaluators (using Groq)...")
    evaluator = DriveWiseEvaluator()

    results = []

    print("\n" + "="*80)
    print("STARTING DRIVEWISE RAG PIPELINE EVALUATION RUNNER")
    print("="*80 + "\n")

    for i, test_case in enumerate(dataset, 1):
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        expected_sources = test_case.get("expected_sources", [])

        print(f"[{i}/{len(dataset)}] Evaluating Query: {question!r}")

        # 1. Retrieval
        start_retrieval = time.perf_counter()
        candidates = chain.retriever.retrieve(question)
        retrieval_time = round((time.perf_counter() - start_retrieval) * 1000, 2)

        # 2. Re-ranking
        start_rerank = time.perf_counter()
        reranked = chain.reranker.rerank(question, candidates, top_n=settings.rerank_top_n)
        reranking_time = round((time.perf_counter() - start_rerank) * 1000, 2)

        # 3. LLM Generation
        start_gen = time.perf_counter()
        # Mirroring chains/rag_chain.py answer() logic
        from app.chains.llm_providers import TemplateAnswerComposer
        if isinstance(chain.llm_provider, TemplateAnswerComposer):
            raw_answer = chain.llm_provider.generate_from_chunks(question, reranked)
        else:
            context_str = format_context(reranked)
            raw_answer = chain._llm_runnable.invoke({"context": context_str, "question": question})
        generation_time = round((time.perf_counter() - start_gen) * 1000, 2)

        total_latency = round(retrieval_time + reranking_time + generation_time, 2)

        print(f"  -> Generated Answer: {raw_answer[:120].replace(chr(10), ' ')}...")
        print(f"  -> Timings: Retrieval={retrieval_time}ms | Rerank={reranking_time}ms | Gen={generation_time}ms | Total={total_latency}ms")

        # 4. Retrieval Metrics (K=5)
        ret_metrics = calculate_retrieval_metrics(reranked, expected_sources, k=5)
        print(f"  -> Retrieval Metrics: Precision@5={ret_metrics['precision_5']} | Recall@5={ret_metrics['recall_5']} | MRR={ret_metrics['mrr']} | Hit={ret_metrics['hit_rate']}")

        # 5. Source Attribution Check (on the top retrieved chunk)
        if reranked:
            top_chunk = reranked[0]
            attrib = validate_source_attribution(
                document_name=top_chunk.metadata.document_name,
                page_number=top_chunk.metadata.page,
                snippet=top_chunk.text,
                generated_answer=raw_answer
            )
            attribution_score = attrib["confidence_score"]
            print(f"  -> Source Attribution: File={top_chunk.metadata.document_name} | Page={top_chunk.metadata.page} | Confidence={attribution_score} ({attrib['reason']})")
        else:
            attribution_score = 0.0
            print("  -> Source Attribution: No sources retrieved.")

        # 6. RAGAS Metrics
        contexts = [c.text for c in reranked]
        print("  -> Computing RAGAS metrics via Groq...")
        ragas_scores = evaluator.evaluate_ragas(question, raw_answer, contexts, expected_answer)
        print(f"     * Faithfulness: {ragas_scores['faithfulness']} | Relevancy: {ragas_scores['answer_relevancy']} | Context Precision: {ragas_scores['context_precision']} | Context Recall: {ragas_scores['context_recall']}")

        # 7. DeepEval Metrics
        print("  -> Computing DeepEval metrics via Groq...")
        deepeval_scores = evaluator.evaluate_deepeval(question, raw_answer, contexts)
        print(f"     * Faithfulness: {deepeval_scores['deepeval_faithfulness']} | Relevancy: {deepeval_scores['deepeval_relevancy']} | Hallucination Score: {deepeval_scores['deepeval_hallucination']}")

        # Compile row
        row = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": raw_answer,
            "retrieved_context": " || ".join(contexts),
            "retrieved_sources": " ; ".join(f"{c.metadata.document_name}:Page {c.metadata.page}" for c in reranked),
            "faithfulness": ragas_scores["faithfulness"],
            "answer_relevancy": ragas_scores["answer_relevancy"],
            "context_precision": ragas_scores["context_precision"],
            "context_recall": ragas_scores["context_recall"],
            "precision_5": ret_metrics["precision_5"],
            "recall_5": ret_metrics["recall_5"],
            "mrr": ret_metrics["mrr"],
            "hit_rate": ret_metrics["hit_rate"],
            "hallucination_score": deepeval_scores["deepeval_hallucination"],
            "retrieval_time": retrieval_time,
            "reranking_time": reranking_time,
            "llm_generation_time": generation_time,
            "latency": total_latency,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Log row to CSV
        log_evaluation_result(row)
        results.append(row)
        print("-" * 80)

    # Summarize run
    if results:
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_prec = sum(r["precision_5"] for r in results) / len(results)
        avg_recall = sum(r["recall_5"] for r in results) / len(results)
        avg_mrr = sum(r["mrr"] for r in results) / len(results)
        avg_hit = sum(r["hit_rate"] for r in results) / len(results)
        avg_halluc = sum(r["hallucination_score"] for r in results) / len(results)
        avg_lat = sum(r["latency"] for r in results) / len(results)

        print("\n" + "="*80)
        print("EVALUATION RUN COMPLETE - SUMMARY OF AVERAGES")
        print("="*80)
        print(f"Average Faithfulness (RAGAS):       {avg_faith:.4f}")
        print(f"Average Answer Relevancy (RAGAS):   {avg_rel:.4f}")
        print(f"Average Precision@5 (Retrieval):    {avg_prec:.4f}")
        print(f"Average Recall@5 (Retrieval):       {avg_recall:.4f}")
        print(f"Average MRR (Retrieval):            {avg_mrr:.4f}")
        print(f"Average Hit Rate:                   {avg_hit:.4f}")
        print(f"Average Hallucination Score (DE):   {avg_halluc:.4f}")
        print(f"Average Total Latency:              {avg_lat:.2f} ms")
        print(f"Results appended to:                {args.output}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
