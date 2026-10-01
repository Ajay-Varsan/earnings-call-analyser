"""
Hybrid Retriever Module: Combining Dense Semantic Search (FAISS)
with Sparse Exact Keyword Matching (BM25) using Reciprocal Rank Fusion (RRF).
Includes persistent disk serialization.
"""

import os
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25."""
    return re.findall(r"\b[a-zA-Z0-9$%\.]+\b", text.lower())


class HybridVectorBM25Retriever:
    """
    Production Hybrid Retriever combining:
    1. Dense Vector Search (Sentence-Transformers all-MiniLM-L6-v2 via FAISS)
    2. Sparse Lexical Search (BM25Okapi)
    3. Reciprocal Rank Fusion (RRF) for ranking harmonization
    """

    def __init__(
        self,
        documents: List[Document],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60
    ):
        self.documents = documents
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.embedding_model_name = embedding_model_name

        # 1. Initialize local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # 2. Build FAISS index
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        # 3. Build BM25 index
        self.corpus_tokens = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        filter_section: Optional[str] = None
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """
        Execute hybrid search and return top_k ranked documents along with
        fused scores and retrieval diagnostics (dense rank vs bm25 rank).
        """
        if not self.documents:
            return []

        # 1. Dense retrieval (fetch top 2*k candidates)
        candidate_k = min(len(self.documents), max(top_k * 3, 15))
        dense_docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=candidate_k
        )

        # 2. Sparse BM25 retrieval
        query_tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:candidate_k]

        # 3. Reciprocal Rank Fusion (RRF) & Weighted Score Fusion
        # RRF formula: RRF_Score = dense_weight * (1 / (rrf_k + dense_rank)) + sparse_weight * (1 / (rrf_k + bm25_rank))
        doc_scores: Dict[int, float] = {}
        doc_diagnostics: Dict[int, Dict[str, Any]] = {}

        # Dense ranks
        for rank, (doc, score) in enumerate(dense_docs_and_scores, start=1):
            chunk_id = doc.metadata.get("chunk_id")
            # find index in self.documents
            doc_idx = next(
                (i for i, d in enumerate(self.documents) if d.metadata.get("chunk_id") == chunk_id),
                None
            )
            if doc_idx is not None:
                dense_rrf = 1.0 / (self.rrf_k + rank)
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + (self.dense_weight * dense_rrf)
                doc_diagnostics.setdefault(doc_idx, {})["dense_rank"] = rank
                doc_diagnostics[doc_idx]["dense_distance"] = float(score)

        # BM25 ranks
        for rank, doc_idx in enumerate(bm25_ranked_indices, start=1):
            if bm25_scores[doc_idx] > 0:
                bm25_rrf = 1.0 / (self.rrf_k + rank)
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + (self.sparse_weight * bm25_rrf)
                doc_diagnostics.setdefault(doc_idx, {})["bm25_rank"] = rank
                doc_diagnostics[doc_idx]["bm25_score"] = float(bm25_scores[doc_idx])

        # Sort aggregated scores descending
        sorted_indices = sorted(doc_scores.keys(), key=lambda i: doc_scores[i], reverse=True)

        # Apply section filtering if requested
        results = []
        for idx in sorted_indices:
            doc = self.documents[idx]
            if filter_section and doc.metadata.get("section_hint") != filter_section:
                continue
            diag = doc_diagnostics.get(idx, {})
            diag["final_hybrid_score"] = doc_scores[idx]
            results.append((doc, doc_scores[idx], diag))
            if len(results) >= top_k:
                break

        return results

    def save_local(self, folder_path: str):
        """Persist hybrid vector and BM25 index to disk."""
        os.makedirs(folder_path, exist_ok=True)
        self.vectorstore.save_local(folder_path)
        with open(os.path.join(folder_path, "bm25_data.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "corpus_tokens": self.corpus_tokens,
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight,
                "rrf_k": self.rrf_k,
                "embedding_model_name": self.embedding_model_name
            }, f)

    @classmethod
    def load_local(cls, folder_path: str, embedding_model_name: str = "all-MiniLM-L6-v2") -> "HybridVectorBM25Retriever":
        """Load persisted hybrid index from disk."""
        with open(os.path.join(folder_path, "bm25_data.pkl"), "rb") as f:
            data = pickle.load(f)

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore = FAISS.load_local(
            folder_path, embeddings, allow_dangerous_deserialization=True
        )

        retriever = cls.__new__(cls)
        retriever.documents = data["documents"]
        retriever.corpus_tokens = data["corpus_tokens"]
        retriever.bm25 = BM25Okapi(data["corpus_tokens"])
        retriever.dense_weight = data.get("dense_weight", 0.6)
        retriever.sparse_weight = data.get("sparse_weight", 0.4)
        retriever.rrf_k = data.get("rrf_k", 60)
        retriever.embedding_model_name = embedding_model_name
        retriever.embeddings = embeddings
        retriever.vectorstore = vectorstore
        return retriever
