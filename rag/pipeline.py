"""
Financial RAG Pipeline using Modern LCEL, Hybrid Retrieval,
Cross-Encoder Reranking, and Strict Grounding Prompts.
"""

from typing import List, Dict, Any, Generator, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from rag.hybrid_retriever import HybridVectorBM25Retriever
from rag.reranker import FlashRankReranker
from rag.financial_analytics import audit_numerical_grounding


FINANCIAL_SYSTEM_PROMPT = """You are an institutional financial analyst assistant.
Your duty is to answer questions regarding earnings calls, SEC filings, and financial statements.

STRICT ACCURACY RULES:
1. Ground every claim ONLY in the provided Context.
2. If a specific metric, date, or fact is not in the context, clearly state: "I could not find this information in the provided documents."
3. Quote exact figures, currency ($), percentages (%), and guidance ranges with precision.
4. Always cite your sources inline using [Source: <doc_name>, Page: <page>].
5. Format comparative data using Markdown tables when appropriate.

Context:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FINANCIAL_SYSTEM_PROMPT),
    ("human", "{question}")
])


class FinancialRAGPipeline:
    """
    End-to-End Enterprise RAG Pipeline for Financial Documents.
    Combines:
    - Hybrid Dense + Sparse BM25 retrieval (Candidate pool = 12)
    - FlashRank neural cross-encoder reranking (Top N = 4)
    - Groq LLaMA-3.1 inference
    - Post-generation quantitative grounding audit
    """

    def __init__(
        self,
        retriever: HybridVectorBM25Retriever,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    ):
        self.retriever = retriever
        self.api_key = api_key
        self.model_name = model_name
        self.reranker = FlashRankReranker(model_name=reranker_model)
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.0,
            api_key=api_key,
            streaming=True
        )
        self.chain = QA_PROMPT | self.llm | StrOutputParser()

    def get_relevant_context(
        self,
        question: str,
        candidate_k: int = 12,
        top_n: int = 4
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """Retrieve hybrid candidates and neural rerank them."""
        candidates = self.retriever.retrieve(query=question, top_k=candidate_k)
        reranked = self.reranker.rerank(query=question, candidates=candidates, top_n=top_n)
        return reranked

    def format_context_string(self, ranked_items: List[Tuple[Document, float, Dict[str, Any]]]) -> str:
        """Format ranked documents into a clean context prompt block with metadata tags."""
        context_parts = []
        for rank, (doc, score, diag) in enumerate(ranked_items, start=1):
            source = doc.metadata.get("source", "Document")
            page = doc.metadata.get("page", 1)
            section = doc.metadata.get("section_hint", "General")
            is_table = doc.metadata.get("is_table", False)
            table_tag = " [Financial Table]" if is_table else ""
            
            header = f"--- [Snippet #{rank} | Source: {source} | Page: {page} | Section: {section}{table_tag} | Relevance: {score:.2f}] ---"
            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def query(self, question: str) -> Dict[str, Any]:
        """Synchronous query execution returning answer, sources, and audit report."""
        ranked_items = self.get_relevant_context(question)
        source_docs = [item[0] for item in ranked_items]
        context_str = self.format_context_string(ranked_items)

        answer = self.chain.invoke({
            "context": context_str,
            "question": question
        })

        # Run numerical grounding audit
        audit = audit_numerical_grounding(answer, source_docs)

        return {
            "question": question,
            "answer": answer,
            "sources": source_docs,
            "ranked_items": ranked_items,
            "audit": audit
        }

    def stream_query(self, question: str) -> Tuple[Generator[str, None, None], List[Tuple[Document, float, Dict[str, Any]]]]:
        """
        Streaming query execution for responsive terminal UX.
        Yields tokens in real-time, returns reranked source items.
        """
        ranked_items = self.get_relevant_context(question)
        context_str = self.format_context_string(ranked_items)

        def token_stream():
            for chunk in self.chain.stream({
                "context": context_str,
                "question": question
            }):
                yield chunk

        return token_stream(), ranked_items
