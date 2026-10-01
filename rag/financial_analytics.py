"""
Financial Analytics Module.
Provides:
1. Automated Executive Scorecard (KPI extraction, Guidance, Bull/Bear thesis)
2. Quantitative Grounding & Fact-Checking Auditor (numerical hallucination detection)
3. Multi-Document Cross-Quarter / Competitor Variance Analysis
"""

import json
import re
from typing import Dict, Any, List, Tuple
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document


def audit_numerical_grounding(answer: str, context_chunks: List[Document]) -> Dict[str, Any]:
    """
    Financial Fact-Checking Guardrail:
    Scans the answer for numbers, currency amounts ($X.XX), and percentages (XX%),
    and verifies whether each figure is grounded in the retrieved context chunks.
    Returns audit statistics and flagging for unverified numbers.
    """
    # Regex to capture currency ($12.5B, $3.20), percentages (15.4%), and standalone metric numbers
    pattern = r"(?i)(\$\s?\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|thousand|[bmkt]\b))?|\b\d+(?:\.\d+)?%)"
    matches = re.findall(pattern, answer)
    unique_figures = list(set([m.strip() for m in matches if len(m.strip()) > 1]))

    all_context = " ".join([c.page_content for c in context_chunks])
    
    verified = []
    unverified = []

    for fig in unique_figures:
        # Normalize comparison: remove spaces, lowercase
        clean_fig = fig.replace(" ", "").replace(",", "").lower()
        clean_context = all_context.replace(" ", "").replace(",", "").lower()

        # Check raw or metric stem
        stem = re.sub(r"[\$%mbtbk]", "", clean_fig)
        if clean_fig in clean_context or (len(stem) >= 2 and stem in clean_context):
            verified.append(fig)
        else:
            unverified.append(fig)

    total_figures = len(unique_figures)
    if total_figures == 0:
        score = 100.0
    else:
        score = round((len(verified) / total_figures) * 100, 1)

    return {
        "grounding_score_pct": score,
        "total_numerical_claims": total_figures,
        "verified_figures": verified,
        "unverified_figures": unverified,
        "is_safe": len(unverified) == 0 or score >= 85.0
    }


def extract_executive_scorecard(
    full_text_or_chunks: List[Document],
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> Dict[str, Any]:
    """
    Extract structured executive KPI scorecard and Bull/Bear takeaways
    using JSON mode inference.
    """
    llm = ChatGroq(
        model=model_name,
        temperature=0.0,
        api_key=api_key,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # Sample top chunks covering overview, tables, and guidance
    sample_texts = []
    total_len = 0
    for doc in full_text_or_chunks[:12]:
        sample_texts.append(f"[Page {doc.metadata.get('page', 1)}]:\n{doc.page_content[:1500]}")
        total_len += len(doc.page_content)
        if total_len > 18000:
            break

    context_prompt = "\n\n---\n\n".join(sample_texts)

    system_prompt = """You are a senior Wall Street equity research analyst.
Extract key financial facts and KPIs from the provided earnings call or financial report.
You must return valid JSON only matching the exact schema below:

{
  "company_name": "Company Name (e.g. Nvidia Corp)",
  "period": "Period (e.g. Q3 FY2025)",
  "financial_kpis": {
    "revenue": "Reported Revenue with YoY %",
    "eps": "Reported EPS (GAAP / Non-GAAP)",
    "gross_margin": "Gross Margin %",
    "operating_income": "Operating Income / Margin %",
    "cash_position": "Cash, equivalents or Free Cash Flow"
  },
  "guidance": {
    "next_quarter_revenue": "Target or range",
    "outlook_summary": "1-2 sentence executive outlook"
  },
  "bull_case_takeaways": [
    "Key positive catalyst 1 with metrics",
    "Key positive catalyst 2 with metrics",
    "Key positive catalyst 3 with metrics"
  ],
  "bear_case_risks": [
    "Key risk or headwind 1 with metrics",
    "Key risk or headwind 2 with metrics",
    "Key risk or headwind 3 with metrics"
  ],
  "executive_sentiment": "Bullish / Neutral / Cautious (with brief explanation)"
}

If any specific metric is not explicitly stated, mark it as "Not Disclosed". Do not invent numbers."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context from earnings document:\n\n{context_prompt}")
    ]

    try:
        response = llm.invoke(messages)
        parsed = json.loads(response.content)
        return parsed
    except Exception as e:
        return {
            "company_name": "Analysis Incomplete",
            "period": "N/A",
            "financial_kpis": {
                "revenue": "Error parsing",
                "eps": "Error parsing",
                "gross_margin": "Error parsing",
                "operating_income": "Error parsing",
                "cash_position": "Error parsing"
            },
            "guidance": {"next_quarter_revenue": "N/A", "outlook_summary": str(e)},
            "bull_case_takeaways": [f"Could not extract: {str(e)}"],
            "bear_case_risks": ["Could not extract"],
            "executive_sentiment": "Undetermined"
        }


def generate_comparative_analysis(
    doc1_name: str,
    doc1_chunks: List[Document],
    doc2_name: str,
    doc2_chunks: List[Document],
    api_key: str,
    comparison_topic: str = "Financial Performance, Guidance & Margin Variance",
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Perform a comparative variance analysis between two documents
    (e.g., Q2 vs Q3, or Competitor A vs Competitor B).
    """
    llm = ChatGroq(
        model=model_name,
        temperature=0.1,
        api_key=api_key
    )

    doc1_sample = "\n\n".join([c.page_content[:1200] for c in doc1_chunks[:6]])
    doc2_sample = "\n\n".join([c.page_content[:1200] for c in doc2_chunks[:6]])

    prompt = f"""You are a CFA charterholder performing a comparative financial analysis.
Compare the following two financial filings/earnings calls on the topic: '{comparison_topic}'.

--- DOCUMENT 1: {doc1_name} ---
{doc1_sample}

--- DOCUMENT 2: {doc2_name} ---
{doc2_sample}

Provide a structured Wall Street research memo in Markdown with:
1. Executive Comparative Summary
2. Variance Table (Metric, {doc1_name}, {doc2_name}, Delta / Variance)
3. Key Strategic Divergences & Trends
4. Risk / Headwind Comparison
Cite exact metrics and numbers from both documents."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
