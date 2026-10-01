"""
Financial Intelligence Terminal - Institutional Earnings Call & 10-K/10-Q Analyser
Powered by Advanced Hybrid RAG (Dense FAISS + Sparse BM25 + FlashRank Cross-Encoder).
"""

import os
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag.document_loader import load_pdf_table_aware, load_pasted_transcript, load_financial_document
from rag.chunking import chunk_financial_documents
from rag.hybrid_retriever import HybridVectorBM25Retriever
from rag.pipeline import FinancialRAGPipeline
from rag.financial_analytics import extract_executive_scorecard, generate_comparative_analysis, audit_numerical_grounding

# Page setup
st.set_page_config(
    page_title="AlphaRAG | Financial Earnings Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Fintech Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }

    /* Header styling */
    .terminal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.4rem;
        background: linear-gradient(90deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    .terminal-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .terminal-badge {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .kpi-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Grounding Audit Badges */
    .grounding-box {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.8rem;
        color: #34d399;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .grounding-box.warning {
        background: rgba(245, 158, 11, 0.08);
        border-color: rgba(245, 158, 11, 0.25);
        color: #fbbf24;
    }

    /* Source Citation Snippets */
    .source-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.84rem;
        line-height: 1.45;
    }
    .source-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #38bdf8;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }

    /* Chat bubble polish */
    .stChatMessage {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        margin-bottom: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State Setup
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_doc_names" not in st.session_state:
    st.session_state.loaded_doc_names = []
if "scorecard" not in st.session_state:
    st.session_state.scorecard = None

# Header Banner
st.markdown("""
<div class="terminal-header">
    <div>
        <div class="terminal-title">⚡ ALPHARAG | Institutional Earnings Terminal</div>
        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 3px;">
            Hybrid Dense-Sparse RAG (FAISS + BM25) • FlashRank Cross-Encoder Reranking • Numerical Fact-Check Guardrails
        </div>
    </div>
    <div>
        <span class="terminal-badge">ENGINE: GROQ LLAMA-3.1</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    env_api_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=env_api_key,
        type="password",
        placeholder="gsk_...",
        help="Get your 100% free API key from https://console.groq.com"
    )

    st.markdown("---")
    st.markdown("### 📂 Document Ingestion")

    ingest_mode = st.radio(
        "Select Source",
        ["📁 Upload Financial Documents (PDF)", "⚡ 1-Click Load Sample Transcript", "📝 Paste Plain Text"],
        index=1
    )

    uploaded_files = []
    pasted_text = ""

    if ingest_mode == "📁 Upload Financial Documents (PDF)":
        uploaded_files = st.file_uploader(
            "Upload 10-K, 10-Q, or Transcripts",
            type=["pdf"],
            accept_multiple_files=True,
            help="Table-aware parser converts financial tables directly into markdown tables."
        )
    elif ingest_mode == "⚡ 1-Click Load Sample Transcript":
        st.info("Loads Nova Semiconductor Q3 FY25 earnings call transcript with segment tables, non-GAAP EPS walk, guidance, and analyst Q&A.")
    else:
        pasted_text = st.text_area("Paste financial text or news", height=140, placeholder="Paste text here...")

    # Retrieval tuning
    with st.expander("⚙️ RAG Engine Hyperparameters"):
        dense_weight = st.slider("Dense Semantic Weight (FAISS)", 0.0, 1.0, 0.6, 0.05)
        sparse_weight = round(1.0 - dense_weight, 2)
        st.caption(f"Sparse Keyword Weight (BM25): **{sparse_weight}**")
        top_k_candidates = st.slider("Hybrid Retrieval Candidates", 6, 20, 12, 2)
        top_n_rerank = st.slider("Cross-Encoder Rerank Cutoff", 2, 8, 4, 1)

    build_btn = st.button("🔨 Index & Build Hybrid Vector Engine", type="primary", use_container_width=True)

    if build_btn:
        if not api_key:
            st.error("Please enter a valid Groq API Key (free from console.groq.com).")
        else:
            with st.spinner("Parsing documents with table preservation and indexing..."):
                try:
                    all_docs = []
                    doc_names = []

                    if ingest_mode == "⚡ 1-Click Load Sample Transcript":
                        sample_path = Path(__file__).parent / "sample_data" / "sample_earnings_call.txt"
                        if sample_path.exists():
                            docs = load_financial_document(str(sample_path))
                            all_docs.extend(docs)
                            doc_names.append("Nova_Semiconductor_Q3_FY25.txt")
                        else:
                            st.error("Sample file not found on disk.")
                    elif ingest_mode == "📁 Upload Financial Documents (PDF)":
                        if not uploaded_files:
                            st.error("Please upload at least one PDF.")
                        else:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                for f in uploaded_files:
                                    fpath = Path(tmpdir) / f.name
                                    fpath.write_bytes(f.read())
                                    parsed = load_pdf_table_aware(str(fpath))
                                    all_docs.extend(parsed)
                                    doc_names.append(f.name)
                    else:
                        if not pasted_text.strip():
                            st.error("Please enter pasted text.")
                        else:
                            docs = load_pasted_transcript(pasted_text, "Pasted_Transcript")
                            all_docs.extend(docs)
                            doc_names.append("Pasted_Transcript")

                    if all_docs:
                        chunks = chunk_financial_documents(all_docs)
                        retriever = HybridVectorBM25Retriever(
                            chunks,
                            dense_weight=dense_weight,
                            sparse_weight=sparse_weight
                        )
                        st.session_state.retriever = retriever
                        st.session_state.all_chunks = chunks
                        st.session_state.loaded_doc_names = doc_names
                        st.session_state.chat_history = []
                        st.session_state.scorecard = None

                        st.success(f"✅ Indexed {len(chunks)} chunks across {len(doc_names)} source(s)!")
                except Exception as e:
                    st.error(f"Error building hybrid index: {str(e)}")

    if st.session_state.loaded_doc_names:
        st.markdown("### 📄 Active Documents")
        for d in st.session_state.loaded_doc_names:
            st.markdown(f"`{d}`")

    if st.session_state.retriever and st.button("🗑️ Reset Engine", use_container_width=True):
        st.session_state.retriever = None
        st.session_state.all_chunks = []
        st.session_state.chat_history = []
        st.session_state.loaded_doc_names = []
        st.session_state.scorecard = None
        st.rerun()

# Main Workspace Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Institutional Q&A Terminal",
    "📊 Executive Scorecard & KPIs",
    "⚖️ Comparative Multi-Doc Analysis",
    "🔬 Hybrid RAG Diagnostics & Explainability"
])

# ----------------- TAB 1: Chat Terminal -----------------
with tab1:
    if not st.session_state.retriever:
        st.info("👈 Please load documents and click **'Index & Build Hybrid Vector Engine'** in the sidebar to begin.")
    else:
        # Predefined prompt chips
        st.markdown("**⚡ Quick Wall Street Prompts:**")
        chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
        selected_prompt = None

        if chip_col1.button("📈 Revenue & Segment Growth", use_container_width=True):
            selected_prompt = "What was the total revenue, YoY growth, and how did each business segment perform?"
        if chip_col2.button("📉 Gross Margin Walk & Guidance", use_container_width=True):
            selected_prompt = "What was reported gross margin, what is the Q4 margin guidance, and why is there compression?"
        if chip_col3.button("💵 Capital Allocation & Buybacks", use_container_width=True):
            selected_prompt = "How much cash did the company return to shareholders and what is the new share repurchase authorization?"
        if chip_col4.button("🔍 Supply Constraints & Yields", use_container_width=True):
            selected_prompt = "What did management state during Q&A regarding packaging capacity, bottlenecks, and production yields?"

        # Display conversation history
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(entry["answer"])

                # Grounding Audit Badge
                audit = entry.get("audit", {})
                if audit:
                    g_score = audit.get("grounding_score_pct", 100.0)
                    is_safe = audit.get("is_safe", True)
                    css_cls = "grounding-box" if is_safe else "grounding-box warning"
                    flag = "🛡️ Verified Grounded" if is_safe else "⚠️ Partial Grounding Discrepancy"
                    st.markdown(
                        f"""<div class="{css_cls}">
                            <b>{flag}</b> | Numerical Confidence: <b>{g_score}%</b> 
                            ({len(audit.get('verified_figures', []))} verified, {len(audit.get('unverified_figures', []))} unverified claims)
                        </div>""",
                        unsafe_allow_html=True
                    )

                # Source inspector expander
                ranked_items = entry.get("ranked_items", [])
                if ranked_items:
                    with st.expander(f"📎 Retrieved & Reranked Context ({len(ranked_items)} Snippets)"):
                        for rank, (doc, score, diag) in enumerate(ranked_items, start=1):
                            src_name = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", 1)
                            sec = doc.metadata.get("section_hint", "General")
                            is_tbl = doc.metadata.get("is_table", False)
                            tag = " [TABLE]" if is_tbl else ""
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-meta">
                                    <span>#{rank} | {src_name} (Page {page}) - {sec}{tag}</span>
                                    <span>Rerank Score: {score:.3f} | BM25 Rank: {diag.get('bm25_rank', 'N/A')}</span>
                                </div>
                                <div style="color: #cbd5e1; font-family: monospace; font-size: 0.82rem; white-space: pre-wrap;">{doc.page_content}</div>
                            </div>
                            """, unsafe_allow_html=True)

        # Chat Input
        user_input = st.chat_input("Ask any quantitative or strategic question about the earnings call...")
        prompt_to_run = selected_prompt or user_input

        if prompt_to_run:
            if not api_key:
                st.error("Please supply your Groq API key in the sidebar.")
            else:
                with st.chat_message("user"):
                    st.write(prompt_to_run)

                with st.chat_message("assistant"):
                    pipeline = FinancialRAGPipeline(
                        retriever=st.session_state.retriever,
                        api_key=api_key
                    )
                    stream_gen, ranked_items = pipeline.stream_query(prompt_to_run)
                    
                    full_answer = st.write_stream(stream_gen)
                    source_docs = [item[0] for item in ranked_items]
                    audit = audit_numerical_grounding(full_answer, source_docs)

                    # Show audit badge
                    g_score = audit.get("grounding_score_pct", 100.0)
                    is_safe = audit.get("is_safe", True)
                    css_cls = "grounding-box" if is_safe else "grounding-box warning"
                    flag = "🛡️ Verified Grounded" if is_safe else "⚠️ Partial Grounding Discrepancy"
                    st.markdown(
                        f"""<div class="{css_cls}">
                            <b>{flag}</b> | Numerical Confidence: <b>{g_score}%</b> 
                            ({len(audit.get('verified_figures', []))} verified, {len(audit.get('unverified_figures', []))} unverified claims)
                        </div>""",
                        unsafe_allow_html=True
                    )

                    st.session_state.chat_history.append({
                        "question": prompt_to_run,
                        "answer": full_answer,
                        "ranked_items": ranked_items,
                        "audit": audit
                    })
                    st.rerun()

# ----------------- TAB 2: Executive Scorecard & KPIs -----------------
with tab2:
    if not st.session_state.retriever:
        st.info("👈 Index a document first to generate the automated executive scorecard.")
    else:
        st.subheader("📑 Automated Executive Scorecard")

        if st.session_state.scorecard is None:
            if st.button("✨ Generate Executive Scorecard & KPI Extraction", type="primary"):
                with st.spinner("Extracting structured financial metrics via JSON Mode LLM..."):
                    scorecard = extract_executive_scorecard(
                        st.session_state.all_chunks,
                        api_key=api_key
                    )
                    st.session_state.scorecard = scorecard
                    st.rerun()

        if st.session_state.scorecard:
            sc = st.session_state.scorecard
            st.markdown(f"### {sc.get('company_name', 'Company')} — {sc.get('period', 'Period')}")
            st.markdown(f"**Executive Sentiment Stance:** `{sc.get('executive_sentiment', 'Neutral')}`")

            # KPI Grid
            kpis = sc.get("financial_kpis", {})
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-label">Reported Revenue</div>
                    <div class="kpi-val" style="color: #38bdf8;">{kpis.get('revenue', 'N/A')}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Diluted EPS</div>
                    <div class="kpi-val" style="color: #34d399;">{kpis.get('eps', 'N/A')}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Gross Margin</div>
                    <div class="kpi-val" style="color: #facc15;">{kpis.get('gross_margin', 'N/A')}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Operating Income</div>
                    <div class="kpi-val" style="color: #a78bfa;">{kpis.get('operating_income', 'N/A')}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Cash / FCF Position</div>
                    <div class="kpi-val" style="color: #f472b6;">{kpis.get('cash_position', 'N/A')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Guidance Banner
            guidance = sc.get("guidance", {})
            with st.container():
                st.markdown(f"""
                <div style="background: #1e1b4b; border-left: 4px solid #818cf8; padding: 12px 16px; border-radius: 6px; margin-bottom: 1.2rem;">
                    <b style="color: #c7d2fe;">Next Quarter Guidance:</b> {guidance.get('next_quarter_revenue', 'N/A')} | 
                    <span style="color: #e0e7ff;">{guidance.get('outlook_summary', '')}</span>
                </div>
                """, unsafe_allow_html=True)

            # Bull vs Bear Columns
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("#### 🟢 Bull Case Catalysts (Tailwinds)")
                for bull in sc.get("bull_case_takeaways", []):
                    st.markdown(f"- {bull}")
            with b_col2:
                st.markdown("#### 🔴 Bear Case Headwinds (Key Risks)")
                for bear in sc.get("bear_case_risks", []):
                    st.markdown(f"- {bear}")

            # Plotly Visualization (Segment Breakdown if Nova sample loaded)
            st.markdown("---")
            st.subheader("📊 Segment Revenue & Margin Walk")
            chart_df = pd.DataFrame({
                "Segment": ["Data Center", "Gaming & PC", "Automotive", "Pro Visualization"],
                "Revenue ($M)": [30771, 3279, 546, 486],
                "YoY Growth (%)": [112.0, 14.8, 63.5, 16.8]
            })
            fig = px.bar(
                chart_df,
                x="Segment",
                y="Revenue ($M)",
                color="YoY Growth (%)",
                color_continuous_scale="Blues",
                title="Q3 Revenue Contribution by Business Segment",
                template="plotly_dark"
            )
            fig.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#111827")
            st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 3: Comparative Analysis -----------------
with tab3:
    st.subheader("⚖️ Multi-Filing Comparative Engine")
    st.markdown("Compare two reports, quarters, or competitors side-by-side to generate a variance memo.")

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        doc1_title = st.text_input("Document 1 Name", value="Q2 FY25 (Prior Quarter)")
        doc1_pasted = st.text_area(
            "Document 1 Text / Summary",
            height=140,
            value="Revenue $30.04B (+122% YoY). Gross margin was 75.7% Non-GAAP. Operating Income $18.64B. Free Cash Flow $13.48B. Data Center revenue was $26.27B. Blackwell packaging bottlenecks flagged at TSMC."
        )
    with comp_col2:
        doc2_title = st.text_input("Document 2 Name", value="Q3 FY25 (Current Quarter)")
        doc2_pasted = st.text_area(
            "Document 2 Text / Summary",
            height=140,
            value="Revenue $35.08B (+93.6% YoY). Gross margin was 75.0% Non-GAAP. Operating Income $21.87B. Free Cash Flow $16.79B. Data Center revenue surged to $30.77B. Guidance for Q4 revenue $37.5B with temporary margin trough at 73.5%."
        )

    topic = st.text_input("Comparison Focus Area", value="Revenue Acceleration, Gross Margin Trough & Guidance Walk")

    if st.button("🚀 Run Comparative Variance Analysis", type="primary"):
        if not api_key:
            st.error("Please provide your Groq API key in the sidebar.")
        else:
            with st.spinner("Synthesizing comparative variance report..."):
                d1_docs = load_pasted_transcript(doc1_pasted, doc1_title)
                d2_docs = load_pasted_transcript(doc2_pasted, doc2_title)
                report = generate_comparative_analysis(
                    doc1_name=doc1_title,
                    doc1_chunks=d1_docs,
                    doc2_name=doc2_title,
                    doc2_chunks=d2_docs,
                    api_key=api_key,
                    comparison_topic=topic
                )
                st.markdown(report)

# ----------------- TAB 4: RAG Diagnostics & Explainability -----------------
with tab4:
    st.subheader("🔬 Hybrid Retrieval Diagnostics & Explainability")
    st.markdown("""
    Inspect how the **Hybrid Retrieval Pipeline** evaluates queries under the hood.
    Demonstrates Reciprocal Rank Fusion (RRF) between **Dense Vector Similarity (FAISS)** 
    and **Sparse Lexical Relevance (BM25)** before cross-encoder reranking.
    """)

    if not st.session_state.retriever:
        st.info("👈 Index a document to inspect retrieval diagnostics.")
    else:
        diag_query = st.text_input("Test Retrieval Query", value="Blackwell yield ramp and gross margin trough")
        if st.button("Inspect Retrieval Ranks"):
            candidates = st.session_state.retriever.retrieve(diag_query, top_k=8)
            
            records = []
            for rank, (doc, score, diag) in enumerate(candidates, start=1):
                records.append({
                    "Hybrid Rank": rank,
                    "Hybrid RRF Score": round(score, 4),
                    "Dense Rank": diag.get("dense_rank", "N/A"),
                    "BM25 Rank": diag.get("bm25_rank", "N/A"),
                    "BM25 Score": round(diag.get("bm25_score", 0.0), 2),
                    "Section Hint": doc.metadata.get("section_hint", "General"),
                    "Is Table": doc.metadata.get("is_table", False),
                    "Snippet Preview": doc.page_content[:120] + "..."
                })
            
            df_diag = pd.DataFrame(records)
            st.dataframe(df_diag, use_container_width=True)
            
            st.caption("Notice how BM25 captures specific technical acronyms while Dense embeddings capture high-level conceptual relevance.")
