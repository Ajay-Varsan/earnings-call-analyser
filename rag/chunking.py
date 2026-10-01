"""
Structure-Preserving Financial Chunking Module.
Splits documents while keeping financial tables, section boundaries,
and numerical disclosures intact.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_financial_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Split financial documents into context-rich chunks while preserving:
    - Markdown table structures
    - Section headers (e.g., Financial Review, Q&A, Guidance)
    - Metadata traceability (page number, chunk sequence, table tags)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n### ",      # Markdown table or section boundary
            "\n\n## ",
            "\n\n",          # Paragraph break
            "\n|",           # Table row start
            "\n",            # Line break
            ". ",            # Sentence break
            " ",
            ""
        ],
        length_function=len
    )

    raw_chunks = splitter.split_documents(documents)
    enriched_chunks: List[Document] = []

    for idx, chunk in enumerate(raw_chunks):
        content = chunk.page_content.strip()
        if not content:
            continue

        is_table = "|" in content and "---" in content
        
        # Detect section keyword heuristics in financial text
        lower_content = content.lower()
        section_hint = "General"
        if "guidance" in lower_content or "outlook" in lower_content:
            section_hint = "Guidance / Outlook"
        elif "question-and-answer" in lower_content or "q&a" in lower_content or "operator" in lower_content:
            section_hint = "Analyst Q&A"
        elif "balance sheet" in lower_content or "cash flow" in lower_content or "income statement" in lower_content:
            section_hint = "Financial Statements"
        elif "revenue" in lower_content and "margin" in lower_content:
            section_hint = "Financial Results"

        metadata = dict(chunk.metadata)
        metadata.update({
            "chunk_id": f"{metadata.get('source', 'doc')}_p{metadata.get('page', 1)}_c{idx}",
            "chunk_index": idx,
            "char_count": len(content),
            "estimated_tokens": max(1, len(content) // 4),
            "is_table": is_table,
            "section_hint": section_hint
        })

        enriched_chunks.append(Document(page_content=content, metadata=metadata))

    return enriched_chunks
