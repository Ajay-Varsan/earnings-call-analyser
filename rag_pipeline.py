"""
Backward-compatibility adapter for legacy imports.
Routes all requests to the modular `rag` package.
"""

from typing import List, Optional
from rag.document_loader import load_financial_document, load_pasted_transcript
from rag.chunking import chunk_financial_documents
from rag.hybrid_retriever import HybridVectorBM25Retriever
from rag.pipeline import FinancialRAGPipeline


def build_vectorstore(pdf_paths: List[str] = None, pasted_text: str = None, api_key: str = None):
    """Legacy wrapper for building the retriever."""
    all_docs = []
    if pdf_paths:
        for p in pdf_paths:
            all_docs.extend(load_financial_document(p))
    if pasted_text:
        all_docs.extend(load_pasted_transcript(pasted_text))

    chunks = chunk_financial_documents(all_docs)
    retriever = HybridVectorBM25Retriever(chunks)
    return retriever


def get_qa_chain(vectorstore: HybridVectorBM25Retriever, api_key: str = None):
    """Legacy wrapper returning pipeline instance."""
    pipeline = FinancialRAGPipeline(retriever=vectorstore, api_key=api_key)
    return pipeline
