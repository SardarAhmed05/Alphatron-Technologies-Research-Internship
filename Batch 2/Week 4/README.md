# 🌐 EDRIC: Autonomous Web Intelligence, Structured Extraction & Real-Time Truth Verification Engine

An advanced, production-grade Multi-Agent Web Intelligence & Extraction system built with **LangGraph**, **LangChain**, and **LLMs (Google Gemini / OpenAI)**. Designed following the modular **5-Step Object-Oriented Pipeline** standard for the **Alphatron Technologies Research Internship (Batch 2 - Week 4)**.

---

## 🌟 Key Features
- **LangGraph Stateful Multi-Agent Architecture**: Stateful cyclic execution graph featuring Web Fetcher, Schema Extractor, Truth Critic, and Intelligence Synthesizer nodes.
- **Cyclic Self-Correction & Reflection Loop**: Automatically detects incomplete schema extractions or low-confidence data, routing back to the extractor with critique feedback.
- **Real-Time Truth & Legitimacy Verification**: Computes a Domain Trust Index (DTI) and Factuality Grounding score (0-100%) to prevent hallucinations and unverified claims.
- **Multi-Format Data Serialization**: Extracts messy web content into clean Pandas DataFrames with 1-click export to CSV, Excel (.xlsx), and JSON.
- **Minimalist Vercel-Style Streamlit UI & Interactive CLI**: Clean dark-mode dashboard and terminal shell ready for 1-click free deployment on Streamlit Community Cloud.
- **5-Step OOP Modular Pipeline**: Structured into 5 self-contained step modules and a master orchestrator (Main.py).
- **Comprehensive 11-Topic Research Manual**: Exhaustive textbook-grade manual covering Transfer Learning, LLM Fine-Tuning, LoRA/QLoRA, Quantization, GPU Mathematics, and Naive vs. Graph RAG.

---

## 📁 Repository Structure
`
Week 4/
├── CASE_STUDY.md                   # 2-Page Executive Case Study & Blueprint
├── Step_1_WebFetcher.py            # Class: WebFetcherPipeline (Live scraping & DOM cleaner)
├── Step_2_ExtractorAgent.py        # Class: SchemaExtractorPipeline (LLM structured entity extraction)
├── Step_3_ValidationGraph.py       # Class: EdricGraphBuilder (LangGraph StateGraph & Truth Critic loop)
├── Step_4_DataExporter.py          # Class: DataExportPipeline (Pandas CSV, Excel, JSON serialization)
├── Step_5_ScraperEvaluator.py      # Class: EdricEvaluatorPipeline (Precision, Trust Score & Latency benchmark)
├── Main.py                         # Class: EdricMasterPipeline (Master OOP pipeline orchestrator)
├── app.py                          # Minimalist Vercel-style Streamlit Web UI
├── cli.py                          # Interactive Terminal CLI
├── generate_manual_pdf.py          # Script generating Report.pdf & RESEARCH_MANUAL.pdf
├── requirements.txt                # Python dependencies
├── .env.example                    # Template environment configuration
├── README.md                       # Installation & Execution Guide
├── TECHNICAL_DOCS.md               # Technical architecture & state specifications
├── USER_MANUAL.md                  # User guide for Web UI & CLI
├── RESEARCH_MANUAL.md              # 11-Topic Deep-Dive Research Manual (Mandatory Task 3 & 4)
├── Report.md                       # Formal internship project report
├── src/                            # Core Source Package
│   ├── config.py                   # Configuration & API Key management
│   ├── state.py                    # LangGraph EdricState TypedDict
│   ├── scraper.py                  # Resilient web scraper & DOM cleaner
│   ├── search.py                   # Live web search integration
│   ├── graph.py                    # LangGraph StateGraph assembler & compiler
│   ├── exporter.py                 # Multi-format CSV/Excel/JSON exporter
│   ├── evaluator.py                # Performance benchmark & evaluation suite
│   └── agents/                     # Specialized Multi-Agent Nodes
│       ├── fetcher_node.py         # Live web & search fetcher node
│       ├── extractor_node.py       # Schema extraction node
│       ├── verifier_node.py        # Truth, Domain Trust & Fact Critic node
│       └── synthesizer_node.py     # Dual format synthesis & table generator node
├── docs/                           # System Diagrams (Mermaid Format)
│   ├── architecture_diagram.md     # System Architecture Diagram
│   ├── flow_diagram.md             # Execution & Scraping Flow Diagram
│   ├── state_diagram.md            # LangGraph State Transition Diagram
│   └── multi_agent_workflow.md     # Multi-Agent Workflow & Reflection Loop
├── sample_data/                    # Out-of-the-box offline test samples
│   ├── sample_ecommerce.html
│   ├── sample_tech_news.html
│   ├── sample_company_profile.html
│   └── generate_samples.py
└── tests/                          # Automated Pytest Suite (11/11 Passed)
    ├── test_scraper.py
    ├── test_extractor.py
    ├── test_verifier.py
    ├── test_graph.py
    └── test_exporter.py
`

---

## 🚀 Installation & Setup Instructions

### 1. Clone the Repository
`ash
git clone https://github.com/SardarAhmed05/Alphatron-Technologies-Research-Internship.git
cd Batch 2/Week 4
`

### 2. Create and Activate Virtual Environment
`ash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
`

### 3. Install Dependencies
`ash
pip install -r requirements.txt
`

### 4. Configure Environment Variables
`ash
cp .env.example .env
`
Edit .env and set your GOOGLE_API_KEY (or OPENAI_API_KEY).

---

## 💻 Execution Guide

### Option A: Run Full Master OOP Pipeline (Main.py)
Executes all 5 pipeline steps sequentially:
`ash
python Main.py sample_data/sample_ecommerce.html
`

### Option B: Launch Streamlit Web UI
`ash
streamlit run app.py
`
Open your browser at http://localhost:8501.

### Option C: Launch Interactive Terminal CLI
`ash
python cli.py
`

### Option D: Run Automated Unit Tests
`ash
pytest -v
`

---

## 📜 License & Credits
Developed by **Sardar Ahmed** for **Alphatron Technologies Research Internship** (Batch 2 - Week 4).