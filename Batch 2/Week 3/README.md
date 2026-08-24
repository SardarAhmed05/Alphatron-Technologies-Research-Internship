# 🤖 Multi-Format Conversational RAG AI Chatbot (OOP Standard)

An intermediate/production-grade Conversational Retrieval-Augmented Generation (RAG) AI Chatbot built using **LangChain**, **ChromaDB**, and **LLMs (Google Gemini / OpenAI / HuggingFace)** with support for **PDF, DOCX, TXT, and Excel/CSV** document formats. Fully refactored into a modular **5-Step Object-Oriented Pipeline** architecture.

---

## 🌟 Key Features

- **5-Step OOP Modular Pipeline**: Structured into 5 self-contained, object-oriented step modules and a master orchestrator (`Main.py`).
- **Multi-Format Document Ingestion**: Ingests PDF (`.pdf`), Word (`.docx`), Plain Text (`.txt`, `.md`), and Excel/CSV (`.xlsx`, `.csv`) files.
- **ChromaDB Vector Persistence**: Stores document embeddings locally in ChromaDB for fast similarity and MMR retrieval.
- **Conversational Memory**: Maintains chat history and uses standalone question rephrasing for multi-turn conversations.
- **Zero-Hallucination Prompting**: System prompt strictly grounds answers in retrieved document context.
- **System Evaluation & Latency Benchmarking**: Evaluates retrieval precision, latency (ms), and context coverage.
- **Dual Interfaces**: Includes an interactive **Streamlit Web UI** and a terminal **CLI**.

---

## 📁 Repository Structure (OOP Standard)

```
Week 3/
├── Step_1_DocumentIngestion.py   # Class: DocumentIngestionPipeline (Multi-format document ingestor & splitter)
├── Step_2_EmbeddingFactory.py    # Class: EmbeddingModelFactory (HuggingFace / Gemini / OpenAI embeddings)
├── Step_3_VectorStoreManager.py   # Class: VectorStoreManager (ChromaDB vector database & search)
├── Step_4_RAGPipeline.py          # Class: ConversationalRAGPipeline (Memory, Prompts, LLM QA chain)
├── Step_5_RAGEvaluator.py         # Class: RAGEvaluationPipeline (Latency, Context Relevance & Benchmark evaluation)
├── Main.py                        # Class: RAGBotMasterPipeline (Master OOP Pipeline Orchestrator)
├── app.py                         # Streamlit Interactive Web Application
├── cli.py                         # Interactive Command-Line Chatbot Interface
├── requirements.txt               # Python dependencies
├── .env.example               # Template environment configuration
├── README.md                  # Installation & Execution Guide
├── TECHNICAL_DOCS.md          # Technical documentation & design rationale
├── src/                       # Core Source Package
│   ├── config.py              # Configuration & Environment Settings
│   ├── loaders.py             # Document Ingestor Implementation
│   ├── embeddings.py          # Embedding Model Factory Implementation
│   ├── vectorstore.py         # ChromaDB Vector Store Implementation
│   ├── prompts.py             # System Prompts & Prompt Templates
│   ├── rag_chain.py           # RAG Pipeline & Memory Chain Implementation
│   └── evaluator.py           # Evaluation Metrics & Benchmarking Implementation
├── docs/                      # System Diagrams (Mermaid Format)
│   ├── architecture_diagram.md # System Architecture Diagram
│   ├── flow_diagram.md         # Document Ingestion & Retrieval Flow Diagram
│   └── state_diagram.md        # Application State Transition Diagram
├── sample_data/               # Out-of-the-box sample test datasets
│   ├── sample_notes.txt
│   ├── sample_data.csv
│   └── generate_samples.py    # Generator script for docx/xlsx sample files
└── tests/                     # OOP Unit Test Suite
    ├── test_loaders.py
    ├── test_vectorstore.py
    ├── test_rag_chain.py
    └── test_evaluator.py
```

---

## 🚀 Installation & Setup Instructions

### Prerequisites
- Python 3.9+ (Python 3.10+ recommended)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/SardarAhmed05/Alphatron-Technologies-Research-Internship.git
cd "Batch 2/Week 3"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API key (if using online LLMs like Google Gemini or OpenAI):
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Optional API Keys for online LLMs
GOOGLE_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Default configuration (HuggingFace local embeddings run 100% locally)
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-flash-latest
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

---

## 💻 Execution Guide

### Option A: Run Full Master OOP Pipeline (`Main.py`)
Executes all 5 pipeline steps sequentially (Ingestion -> Embeddings -> VectorStore -> RAG QA -> Evaluation):
```bash
python Main.py
```

### Option B: Run Individual Pipeline Steps (OOP)
Each step can be executed as a standalone OOP module:
```bash
python Step_1_DocumentIngestion.py
python Step_2_EmbeddingFactory.py
python Step_3_VectorStoreManager.py
python Step_4_RAGPipeline.py
python Step_5_RAGEvaluator.py
```

### Option C: Launch Streamlit Web UI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can upload documents, process them into ChromaDB, ask questions, inspect source citations, and manage database state visually.

### Option D: Launch Interactive Terminal CLI
```bash
python cli.py
```
Supported CLI commands:
- `/ingest <file_or_folder>`: Ingest documents into ChromaDB.
- `/clear`: Clear the vector database index.
- `/reset_memory`: Reset chat history.
- `exit`: Quit the chatbot.

---

## 🧪 Running Automated Unit Tests

Run the test suite using `pytest`:
```bash
pytest -v
```

---

## 📊 System Diagrams

Full Mermaid diagrams are provided in the `docs/` folder:
1. **[Architecture Diagram](docs/architecture_diagram.md)**: Visualizes the 5-layer modular OOP system design.
2. **[Flow Diagram](docs/flow_diagram.md)**: Details sequence steps for document chunking, embedding, vector search, and LLM prompt generation.
3. **[State Diagram](docs/state_diagram.md)**: Shows state transitions during indexing, retrieval, and memory management.

---

## 📜 License & Credits
Developed for **Alphatron Technologies Research Internship** (Batch 2 - Week 3).
