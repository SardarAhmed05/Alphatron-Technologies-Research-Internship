# 🤖 Multi-Format Conversational RAG AI Chatbot

An intermediate/production-grade Conversational Retrieval-Augmented Generation (RAG) AI Chatbot built using **LangChain**, **ChromaDB**, and **LLMs (Google Gemini / OpenAI / HuggingFace)** with support for **PDF, DOCX, TXT, and Excel/CSV** document formats.

---

## 🌟 Key Features

- **Multi-Format Document Ingestion**: Ingests PDF (`.pdf`), Word (`.docx`), Plain Text (`.txt`, `.md`), and Excel/CSV (`.xlsx`, `.csv`) files.
- **ChromaDB Vector Persistence**: Stores document embeddings locally in ChromaDB for fast similarity retrieval.
- **Conversational Memory**: Maintains chat history and uses standalone question rephrasing for multi-turn conversations.
- **Zero-Hallucination Prompting**: System prompt strictly grounds answers in retrieved document context.
- **Source Attribution**: Displays exact document citations (file name, page number, sheet name, snippet preview) for every response.
- **Dual Interfaces**: Includes an interactive **Streamlit Web UI** and a terminal **CLI**.

---

## 📁 Repository Structure

```
Week 3/
├── app.py                     # Streamlit Interactive Web Application
├── cli.py                     # Interactive Command-Line Chatbot Interface
├── main.py                    # Programmatic API & Script Entry Point
├── requirements.txt           # Python dependencies
├── .env.example               # Template environment configuration
├── README.md                  # Installation & Execution Guide
├── TECHNICAL_DOCS.md          # Technical documentation & design rationale
├── src/                       # Core Source Code Module
│   ├── config.py              # Configuration & Environment Settings
│   ├── loaders.py             # Multi-Format Document Ingestor (PDF, DOCX, TXT, Excel/CSV)
│   ├── embeddings.py          # Embedding Model Factory (HuggingFace / Gemini / OpenAI)
│   ├── vectorstore.py         # ChromaDB Vector Database Manager
│   ├── prompts.py             # System Prompts & Prompt Engineering Templates
│   └── rag_chain.py           # Conversational RAG Pipeline & Memory Chain
├── docs/                      # System Diagrams (Mermaid Format)
│   ├── architecture_diagram.md # System Architecture Diagram
│   ├── flow_diagram.md         # Document Ingestion & Retrieval Flow Diagram
│   └── state_diagram.md        # Application State Transition Diagram
├── sample_data/               # Out-of-the-box sample test datasets
│   ├── sample_notes.txt
│   ├── sample_data.csv
│   └── generate_samples.py    # Generator script for docx/xlsx sample files
└── tests/                     # Unit Tests Suite
    ├── test_loaders.py
    ├── test_vectorstore.py
    └── test_rag_chain.py
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

### Option A: Launch Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can upload documents, process them into ChromaDB, ask questions, inspect source citations, and manage database state visually.

### Option B: Launch Interactive Terminal CLI
```bash
python cli.py
```
Supported CLI commands:
- `/ingest <file_or_folder>`: Ingest documents into ChromaDB.
- `/clear`: Clear the vector database index.
- `/reset_memory`: Reset chat history.
- `exit`: Quit the chatbot.

### Option C: Run One-off Ingestion & Query Script
```bash
# Ingest a document directory
python main.py --ingest ./sample_data

# Ask a question
python main.py --query "Who is the lead RAG architect?"
```

---

## 🧪 Running Automated Unit Tests

Run the test suite using `pytest`:
```bash
pytest -v
```

---

## 📊 System Diagrams

Full Mermaid diagrams are provided in the `docs/` folder:
1. **[Architecture Diagram](docs/architecture_diagram.md)**: Visualizes the 5-layer modular system design.
2. **[Flow Diagram](docs/flow_diagram.md)**: Details sequence steps for document chunking, embedding, vector search, and LLM prompt generation.
3. **[State Diagram](docs/state_diagram.md)**: Shows state transitions during indexing, retrieval, and memory management.

---

## 📜 License & Credits
Developed for **Alphatron Technologies Research Internship** (Batch 2 - Week 3).
