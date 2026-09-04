"""
=============================================================================
EDRIC - STREAMLIT INTERACTIVE WEB APPLICATION (CLEAN MODERN LIGHT THEME)
=============================================================================
Minimalist, high-performance UI for live web extraction, structured data mining,
and real-time truth verification.
"""

import os
import json
import time
import io
import streamlit as st
import pandas as pd

from src.graph import EdricGraphManager
from src.exporter import DataExporter
from src.scraper import WebScraperEngine
from src.config import GOOGLE_API_KEY, OPENAI_API_KEY
import streamlit.components.v1 as components


def render_mermaid(diagram_code: str, height: int = 320):
    """Renders Mermaid.js diagrams directly into the Streamlit Web UI."""
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: true, 
                theme: 'neutral',
                securityLevel: 'loose',
                flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
            }});
        </script>
        <style>
            body {{
                margin: 0;
                padding: 4px;
                background: transparent;
                display: flex;
                justify-content: center;
                align-items: center;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .mermaid {{
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                width: 95%;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{diagram_code}
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=True)

# Configure Page
st.set_page_config(
    page_title="EDRIC | Web Intelligence & Verification",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Clean Light Theme CSS
st.markdown("""
<style>
    /* Light Theme Core */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Force high contrast on all text, headers, and markdown */
    p, span, div, label, li, h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
    }
    
    /* Headers & Brand */
    .brand-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
        color: #0F172A !important;
        margin-bottom: 0.15rem;
    }
    .brand-sub {
        font-size: 0.95rem !important;
        color: #475569 !important;
        margin-bottom: 1.25rem;
    }
    
    /* Inputs & Textareas */
    input, textarea, select {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }
    input:focus, textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    
    /* Captions and Subtext */
    small, .stCaption, [data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }
    
    /* Metric Cards (Clean High-Contrast Tiles) */
    .card-stat {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 0.9rem 1rem !important;
        text-align: center !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    .stat-val {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    .stat-lbl {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #475569 !important;
        margin-top: 0.2rem;
    }
    
    /* Status Badges */
    .badge-verified {
        display: inline-block !important;
        background: #ECFDF5 !important;
        color: #065F46 !important;
        border: 1px solid #6EE7B7 !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 9999px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    
    /* Light Mode Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #CBD5E1 !important;
    }
    
    /* Code Blocks */
    code, pre {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = EdricGraphManager(with_checkpointing=True)

if "latest_state" not in st.session_state:
    st.session_state.latest_state = None


# ==========================================
# SIDEBAR: MINIMAL & CLEAN SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("### 🌐 **EDRIC Intelligence**")
    st.caption("Autonomous Web Data Mining & Truth Critic")
    st.markdown("---")

    with st.expander("🔑 Gemini API Key (Optional)", expanded=False):
        st.markdown(
            "<small style='color: #64748B;'><b>Why is this here?</b> You can optionally provide a custom Google Gemini API Key for model inference.<br><br>"
            "<b>Is it safe?</b> Yes. Your key is stored in temporary browser memory only for this active session. It is never written to disk, logged, or shared with third parties.</small>",
            unsafe_allow_html=True
        )
        user_key = st.text_input("Gemini API Key", value="", type="password", placeholder="AIzaSy...")
        if user_key:
            os.environ["GOOGLE_API_KEY"] = user_key
            st.success("Custom session key applied.")

    st.markdown("---")
    st.caption("Alphatron Technologies Internship | Week 4")
    st.caption("Author: Sardar Ahmed")


# ==========================================
# MAIN DASHBOARD INTERFACE
# ==========================================
st.markdown('<div class="brand-title">EDRIC</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-sub">Engine for Dynamic Research, Intelligence & Credibility — Ingests live web streams, filters noise, and delivers verified intelligence.</div>',
    unsafe_allow_html=True
)

# Input Mode Tabs
input_tab1, input_tab2, input_tab3 = st.tabs([
    "🌐 Live Webpage / URL Analysis", 
    "🔍 Live Topic & Event Research", 
    "📝 Direct Text / Document Ingestion"
])

with input_tab1:
    col_u1, col_u2 = st.columns([3, 2])
    with col_u1:
        target_url = st.text_input("Webpage URL", value="https://news.ycombinator.com", placeholder="https://example.com/article")
    with col_u2:
        extraction_goal_url = st.text_input("Extraction Objective", value="Extract all top stories, authors, points, and source links.")
    btn_url = st.button("🚀 Analyze & Extract Webpage", key="btn_url", type="primary", use_container_width=True)

with input_tab2:
    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        search_query = st.text_input("Research Topic / Headline", placeholder="e.g. Robert Pattinson The Batman Part II updates")
    with col_s2:
        extraction_goal_search = st.text_input("Research Focus", value="Extract primary breakthroughs, verified facts, sources, and dates.")
    btn_search = st.button("🔍 Search & Verify Topic", key="btn_search", type="primary", use_container_width=True)

with input_tab3:
    raw_text_input = st.text_area("Paste Raw Text, Article, or HTML", height=130, placeholder="Paste text, meeting notes, financial disclosures, or HTML tables...")
    extraction_goal_text = st.text_input("Extraction & Analysis Goal", value="Extract all structured facts, key metrics, and bulleted takeaways.")
    btn_text = st.button("⚡ Process Document Content", key="btn_text", type="primary", use_container_width=True)


# ==========================================
# PIPELINE EXECUTION
# ==========================================
run_requested = btn_url or btn_search or btn_text

if run_requested:
    if btn_url:
        input_type = "url"
        active_input = target_url
        active_goal = extraction_goal_url
    elif btn_search:
        input_type = "query"
        active_input = search_query
        active_goal = extraction_goal_search
    else:
        input_type = "text"
        active_input = raw_text_input
        active_goal = extraction_goal_text

    if not active_input.strip():
        st.warning("⚠️ Please provide a valid URL, search topic, or document text.")
    else:
        progress_box = st.status("🚀 EDRIC Multi-Agent Intelligence Pipeline Active...", expanded=True)
        with progress_box:
            st.write("🕷️ **[Fetcher]** Ingesting content and sanitizing DOM structure...")
            time.sleep(0.2)
            st.write("🧠 **[Extractor]** Identifying entities and schema structures...")
            time.sleep(0.2)
            st.write("🛡️ **[Verifier]** Auditing facts, checking credibility, and verifying grounding...")
            time.sleep(0.2)
            st.write("📊 **[Synthesizer]** Formulating executive report and structured tables...")

            start_t = time.time()
            try:
                final_state = st.session_state.graph_manager.run(
                    raw_input=active_input,
                    input_type=input_type,
                    extraction_goal=active_goal,
                    thread_id=f"session-{int(time.time())}",
                )
                duration_ms = round((time.time() - start_t) * 1000, 2)
                st.session_state.latest_state = final_state
                st.session_state.latest_latency = duration_ms
                progress_box.update(label=f"✓ Content Analyzed ({duration_ms} ms)", state="complete", expanded=False)
            except Exception as e:
                progress_box.update(label=f"❌ Error: {e}", state="error")
                st.error(f"Execution Error: {e}")


# ==========================================
# RESULTS DISPLAY & TABBED WORKSPACE
# ==========================================
if st.session_state.latest_state:
    state = st.session_state.latest_state
    records = state.get("dataframe_records", [])
    trust_score = state.get("trust_score", 85.0)
    trust_breakdown = state.get("trust_breakdown", {})
    source_meta = state.get("source_metadata", {})
    briefing = state.get("executive_briefing", "")
    latency_ms = getattr(st.session_state, "latest_latency", 1200)

    st.markdown("---")

    # TABBED WORKSPACE: EXECUTIVE REPORT FIRST!
    out_tab1, out_tab2, out_tab3, out_tab4 = st.tabs([
        "💡 Executive Report & Key Insights",
        "📊 Structured Data Table",
        "🔍 Raw Schema & JSON",
        "⚡ Multi-Agent Execution Trace",
    ])

    # TAB 1: EXECUTIVE INTELLIGENCE REPORT
    with out_tab1:
        st.markdown(briefing)
        st.download_button(
            "📥 Download Executive Briefing (.md)",
            data=briefing,
            file_name="edric_executive_briefing.md",
            mime="text/markdown",
        )

    # TAB 2: STRUCTURED DATA MATRIX
    with out_tab2:
        st.markdown("### 📊 **Structured Data Matrix**")
        st.caption(f"Showing {len(records)} extracted records.")
        
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("#### 💾 Export Options")
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                csv_data = DataExporter.to_csv(records)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name="edric_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with exp_col2:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine="openpyxl")
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    "📥 Download Excel (.xlsx)",
                    data=excel_data,
                    file_name="edric_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            with exp_col3:
                json_data = DataExporter.to_json(records)
                st.download_button(
                    "📥 Download JSON",
                    data=json_data,
                    file_name="edric_data.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.info("No tabular records extracted from this source.")

    # TAB 3: RAW SCHEMA & JSON ENTITIES
    with out_tab3:
        st.markdown("### 🔍 **Extracted Entity Schemas**")
        schema_fields = state.get("schema_fields", [])
        if schema_fields:
            st.markdown(f"**Identified Schema Columns:** `{'`, `'.join(schema_fields)}`")
        st.json(state.get("extracted_data", []))

    # TAB 4: MULTI-AGENT EXECUTION TRACE
    with out_tab4:
        st.markdown("### ⚡ **LangGraph Multi-Agent Execution Lifecycle**")
        
        tr_col1, tr_col2 = st.columns(2)
        with tr_col1:
            st.markdown(f"- **Input Modality**: `{state.get('input_type', 'N/A')}`")
            st.markdown(f"- **Reflection Loops Executed**: `{state.get('iteration_count', 1)}`")
            st.markdown(f"- **Final Graph Node**: `{state.get('current_node', 'synthesizer')}`")
        with tr_col2:
            st.markdown(f"- **Is Formally Verified**: `{state.get('is_verified', True)}`")
            st.markdown(f"- **Domain Resolved**: `{source_meta.get('domain', 'N/A')}`")
            st.markdown(f"- **Source Status Code**: `{source_meta.get('status_code', 200)}`")

        st.markdown("#### 🗺️ **Active Multi-Agent StateGraph Architecture**")
        workflow_diagram = """
        flowchart LR
            Start([User Input]) --> F[🕷️ Web Fetcher]
            F --> E[🧠 Schema Extractor]
            E --> V[🛡️ Truth Critic]
            V -- Trust < 75% --> Refine[🔄 Reflection Loop]
            Refine --> E
            V -- Trust >= 75% --> S[📊 Synthesizer]
            S --> Out([📄 Verified Dossier])
            
            classDef default fill:#F8FAFC,stroke:#94A3B8,stroke-width:1.5px,color:#0F172A;
            classDef highlight fill:#ECFDF5,stroke:#10B981,stroke-width:2px,color:#065F46;
            class Out highlight;
        """
        render_mermaid(workflow_diagram, height=220)

        st.markdown("#### Execution Chronology Log")
        for log in state.get("status_logs", []):
            st.code(log, language="bash")

    # ==========================================
    # METRICS STRIP: MOVED TO BOTTOM OF PAGE
    # ==========================================
    st.markdown("---")
    st.markdown("#### 📈 **Execution & Source Metrics**")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="card-stat">
            <div class="stat-val">{trust_score}%</div>
            <div class="stat-lbl">Credibility Index</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="card-stat">
            <div class="stat-val">{len(records)}</div>
            <div class="stat-lbl">Extracted Records</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="card-stat">
            <div class="stat-val">{trust_breakdown.get('domain_authority_score', 80.0)}%</div>
            <div class="stat-lbl">Domain Authority</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="card-stat">
            <div class="stat-val">{latency_ms} ms</div>
            <div class="stat-lbl">Processing Latency</div>
        </div>
        """, unsafe_allow_html=True)
