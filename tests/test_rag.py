"""
Unit tests for the Financial RAG pipeline components.
"""

import pytest
from langchain_core.documents import Document
from rag.document_loader import load_pasted_transcript, _table_to_markdown
from rag.chunking import chunk_financial_documents
from rag.hybrid_retriever import HybridVectorBM25Retriever
from rag.financial_analytics import audit_numerical_grounding


def test_table_to_markdown():
    sample_table = [
        ["Metric", "Q3 FY25", "YoY"],
        ["Revenue", "$35.1B", "+94%"],
        ["EPS", "$0.81", "+102%"]
    ]
    md = _table_to_markdown(sample_table)
    assert "| Metric | Q3 FY25 | YoY |" in md
    assert "| --- | --- | --- |" in md
    assert "| Revenue | $35.1B | +94% |" in md


def test_financial_chunking():
    doc_text = """# Company Q3 Results
Here are the prepared remarks for the quarter. Revenue was strong across all segments.

### Extracted Financial Tables:
| Segment | Revenue |
| --- | --- |
| Data Center | $30.8B |
| Gaming | $3.2B |

### OUTLOOK FOR Q4:
Management expects continued strength with guidance of $37.5B."""

    doc = Document(page_content=doc_text, metadata={"source": "test_doc.txt", "page": 1})
    chunks = chunk_financial_documents([doc], chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "estimated_tokens" in chunk.metadata
        assert "section_hint" in chunk.metadata

    # Verify table is identified
    table_chunks = [c for c in chunks if c.metadata.get("is_table")]
    assert len(table_chunks) >= 1


def test_hybrid_retriever():
    sample_docs = [
        Document(
            page_content="Data Center revenue reached $30.8 billion, representing 88% of total revenue.",
            metadata={"chunk_id": "c1", "source": "sample.txt", "page": 1, "section_hint": "Financial Results"}
        ),
        Document(
            page_content="Automotive and robotics revenue was $546 million, up 63.5% year-over-year.",
            metadata={"chunk_id": "c2", "source": "sample.txt", "page": 1, "section_hint": "Financial Results"}
        ),
        Document(
            page_content="Capital expenditures were $3.1 billion and free cash flow totaled $16.8 billion.",
            metadata={"chunk_id": "c3", "source": "sample.txt", "page": 2, "section_hint": "Balance Sheet"}
        ),
    ]

    retriever = HybridVectorBM25Retriever(sample_docs)
    results = retriever.retrieve("What was the Data Center revenue?", top_k=2)

    assert len(results) == 2
    top_doc, score, diag = results[0]
    assert "Data Center" in top_doc.page_content
    assert "bm25_rank" in diag or "dense_rank" in diag


def test_numerical_grounding_auditor():
    context_chunks = [
        Document(page_content="Total revenue reached $35.1 billion, representing 94% growth.")
    ]

    # Test 1: Fully grounded claims
    grounded_answer = "The company reported total revenue of $35.1 billion, reflecting a 94% increase."
    audit_pass = audit_numerical_grounding(grounded_answer, context_chunks)
    assert audit_pass["grounding_score_pct"] == 100.0
    assert audit_pass["is_safe"] is True

    # Test 2: Ungrounded / hallucinated claim
    hallucinated_answer = "The revenue reached $48.5 billion with a 25% margin."
    audit_fail = audit_numerical_grounding(hallucinated_answer, context_chunks)
    assert audit_fail["grounding_score_pct"] < 50.0
    assert "$48.5 billion" in audit_fail["unverified_figures"]
