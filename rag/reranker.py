"""
Cross-Encoder Reranker Module using FlashRank.
Provides zero-cost, high-speed neural reranking to boost Context Precision
and filter out irrelevant financial noise before generation.
"""

from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest


class FlashRankReranker:
    """
    Local Cross-Encoder Reranker.
    Uses an ONNX-optimized transformer to score cross-attention between
    the user query and candidate document chunks.
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self.ranker = Ranker(model_name=model_name, cache_dir=".cache/flashrank")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float, Dict[str, Any]]],
        top_n: int = 4
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """
        Rerank retrieved candidates based on cross-encoder similarity score.
        Returns top_n documents with rerank scores and updated diagnostics.
        """
        if not candidates:
            return []

        passages = []
        for idx, (doc, hybrid_score, diag) in enumerate(candidates):
            passages.append({
                "id": idx,
                "text": doc.page_content,
                "meta": {
                    "doc": doc,
                    "hybrid_score": hybrid_score,
                    "diag": diag
                }
            })

        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_req)

        final_ranked: List[Tuple[Document, float, Dict[str, Any]]] = []
        for rank, res in enumerate(ranked_results[:top_n], start=1):
            original_meta = res.get("meta", {})
            doc = original_meta.get("doc")
            diag = dict(original_meta.get("diag", {}))
            
            rerank_score = float(res.get("score", 0.0))
            diag["rerank_rank"] = rank
            diag["rerank_score"] = round(rerank_score, 4)

            final_ranked.append((doc, rerank_score, diag))

        return final_ranked
