"""
Document Loader module with Table-Aware Financial PDF extraction.
Uses pdfplumber to convert financial tables into Markdown format,
preserving critical tabular relationships (revenue, margins, balance sheet line items).
"""

import os
from pathlib import Path
from typing import List, Optional
import pdfplumber
from langchain_core.documents import Document
from pypdf import PdfReader


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """Convert a 2D table grid extracted by pdfplumber into Markdown table syntax."""
    if not table or len(table) < 2:
        return ""
    
    # Filter empty rows and sanitize cells
    cleaned_rows = []
    for row in table:
        cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)
            
    if len(cleaned_rows) < 2:
        return ""
        
    num_cols = max(len(row) for row in cleaned_rows)
    # Normalize row lengths
    normalized_rows = [row + [""] * (num_cols - len(row)) for row in cleaned_rows]
    
    header = normalized_rows[0]
    separator = ["---"] * num_cols
    body = normalized_rows[1:]
    
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |"
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
        
    return "\n".join(lines)


def load_pdf_table_aware(pdf_path: str) -> List[Document]:
    """
    Extract text and tables from a PDF document with structure preservation.
    Financial tables are converted into Markdown tables so numbers remain aligned.
    """
    documents: List[Document] = []
    file_name = Path(pdf_path).name

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # 1. Extract tables
                tables = page.extract_tables()
                md_tables = []
                for table in tables:
                    md_t = _table_to_markdown(table)
                    if md_t:
                        md_tables.append(md_t)
                
                # 2. Extract plain text
                raw_text = page.extract_text() or ""
                
                # If tables were detected, append structured markdown table representation
                content_parts = []
                if raw_text.strip():
                    content_parts.append(raw_text.strip())
                if md_tables:
                    content_parts.append("\n\n### Extracted Financial Tables:\n" + "\n\n".join(md_tables))
                    
                full_content = "\n\n".join(content_parts)
                if full_content.strip():
                    documents.append(
                        Document(
                            page_content=full_content,
                            metadata={
                                "source": file_name,
                                "file_path": str(pdf_path),
                                "page": page_idx,
                                "has_tables": len(md_tables) > 0,
                                "table_count": len(md_tables)
                            }
                        )
                    )
    except Exception as e:
        # Fallback to PyPDF if pdfplumber fails on malformed fonts or scanned pages
        print(f"[DocumentLoader] pdfplumber notice: {e}, attempting pypdf fallback for {file_name}")
        reader = PdfReader(pdf_path)
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": file_name,
                            "file_path": str(pdf_path),
                            "page": page_idx,
                            "has_tables": False,
                            "table_count": 0
                        }
                    )
                )

    return documents


def load_financial_document(file_path: str) -> List[Document]:
    """Load financial document based on extension (PDF or plain text)."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf_table_aware(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [
            Document(
                page_content=text,
                metadata={
                    "source": Path(file_path).name,
                    "file_path": str(file_path),
                    "page": 1,
                    "has_tables": False,
                    "table_count": 0
                }
            )
        ]


def load_pasted_transcript(text: str, source_name: str = "Pasted_Transcript") -> List[Document]:
    """Wrap raw pasted text into a LangChain Document with metadata."""
    return [
        Document(
            page_content=text.strip(),
            metadata={
                "source": source_name,
                "file_path": source_name,
                "page": 1,
                "has_tables": "|" in text,
                "table_count": 1 if "|" in text else 0
            }
        )
    ]
