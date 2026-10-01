"""
Financial Intelligence RAG Engine
Modular package for earnings call and financial document analysis.
"""

from rag.document_loader import load_financial_document, load_pasted_transcript
from rag.chunking import chunk_financial_documents
from rag.hybrid_retriever import HybridVectorBM25Retriever
from rag.reranker import FlashRankReranker
from rag.pipeline import FinancialRAGPipeline
from rag.financial_analytics import extract_executive_scorecard, generate_comparative_analysis

__all__ = [
    "load_financial_document",
    "load_pasted_transcript",
    "chunk_financial_documents",
    "HybridVectorBM25Retriever",
    "FlashRankReranker",
    "FinancialRAGPipeline",
    "extract_executive_scorecard",
    "generate_comparative_analysis",
]
