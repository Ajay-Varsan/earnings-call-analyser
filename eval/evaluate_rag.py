"""
Automated RAG Evaluation Suite.
Benchmarks the Financial RAG Pipeline on Context Recall,
Answer Keyword Precision, Numerical Grounding, and Latency.
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.document_loader import load_financial_document
from rag.chunking import chunk_financial_documents
from rag.hybrid_retriever import HybridVectorBM25Retriever
from rag.pipeline import FinancialRAGPipeline


def run_benchmark(api_key: str, dataset_path: str = None, sample_path: str = None) -> Dict[str, Any]:
    """Execute the full benchmark suite on the golden dataset."""
    root_dir = Path(__file__).parent.parent
    if dataset_path is None:
        dataset_path = str(root_dir / "eval" / "golden_dataset.json")
    if sample_path is None:
        sample_path = str(root_dir / "sample_data" / "sample_earnings_call.txt")

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"\n=======================================================")
    print(f"🚀 FINANCIAL RAG BENCHMARK EVALUATION SUITE")
    print(f"=======================================================")
    print(f"[*] Loading sample document: {Path(sample_path).name}")

    raw_docs = load_financial_document(sample_path)
    chunks = chunk_financial_documents(raw_docs)
    print(f"[*] Processed {len(raw_docs)} document(s) into {len(chunks)} structured chunks.")

    print(f"[*] Building Hybrid Retriever (FAISS + BM25)...")
    retriever = HybridVectorBM25Retriever(chunks)
    pipeline = FinancialRAGPipeline(retriever=retriever, api_key=api_key)

    total_queries = len(golden_data)
    results: List[Dict[str, Any]] = []

    retrieval_hits = 0
    generation_hits = 0
    total_grounding_score = 0.0
    total_latency = 0.0

    print(f"\n[*] Executing {total_queries} evaluation queries...\n")

    for item in golden_data:
        qid = item["id"]
        q = item["question"]
        expected_keywords = item["expected_keywords"]

        start_t = time.perf_counter()
        query_result = pipeline.query(q)
        latency = time.perf_counter() - start_t
        total_latency += latency

        answer = query_result["answer"]
        ranked_items = query_result["ranked_items"]
        retrieved_text = " ".join([item[0].page_content for item in ranked_items])
        audit = query_result["audit"]

        # Check retrieval recall (are keywords in retrieved chunks?)
        r_hit_count = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text.lower())
        r_recall = r_hit_count / len(expected_keywords)
        if r_recall >= 0.7:
            retrieval_hits += 1

        # Check generation accuracy (are keywords in the answer?)
        g_hit_count = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
        g_precision = g_hit_count / len(expected_keywords)
        if g_precision >= 0.7:
            generation_hits += 1

        total_grounding_score += audit["grounding_score_pct"]

        print(f"[{qid}] Latency: {latency:.2f}s | Retrieval Recall: {r_recall*100:.0f}% | Answer Score: {g_precision*100:.0f}% | Grounding: {audit['grounding_score_pct']}%")

        results.append({
            "id": qid,
            "question": q,
            "expected_keywords": expected_keywords,
            "retrieval_recall_pct": round(r_recall * 100, 1),
            "answer_precision_pct": round(g_precision * 100, 1),
            "grounding_score_pct": audit["grounding_score_pct"],
            "latency_seconds": round(latency, 2),
            "generated_answer": answer
        })

    avg_retrieval_recall = round((retrieval_hits / total_queries) * 100, 1)
    avg_gen_accuracy = round((generation_hits / total_queries) * 100, 1)
    avg_grounding = round(total_grounding_score / total_queries, 1)
    avg_latency = round(total_latency / total_queries, 2)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": total_queries,
        "retrieval_recall_success_rate": f"{avg_retrieval_recall}%",
        "answer_accuracy_success_rate": f"{avg_gen_accuracy}%",
        "average_grounding_confidence": f"{avg_grounding}%",
        "average_latency_seconds": avg_latency,
        "detailed_results": results
    }

    print(f"\n=======================================================")
    print(f"📊 BENCHMARK SUMMARY REPORT")
    print(f"=======================================================")
    print(f"Total Evaluated Queries:       {total_queries}")
    print(f"Retrieval Recall Hit Rate:     {avg_retrieval_recall}%")
    print(f"Answer Fact Precision:         {avg_gen_accuracy}%")
    print(f"Average Grounding Confidence:  {avg_grounding}%")
    print(f"Average Latency:               {avg_latency}s / query")
    print(f"=======================================================\n")

    output_path = root_dir / "eval" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Detailed report saved to: {output_path}")

    return summary


if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[!] Error: GROQ_API_KEY environment variable not found. Pass it or set it in .env.")
        sys.exit(1)
    run_benchmark(api_key=api_key)
