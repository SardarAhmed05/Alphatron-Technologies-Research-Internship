# 📖 USER & DEVELOPER OPERATIONAL MANUAL
## Multi-Format Conversational RAG AI Chatbot

---

## 📑 TABLE OF CONTENTS
1. [System Requirements](#1-system-requirements)
2. [Installation & Setup](#2-installation--setup)
3. [Environment Configuration (.env)](#3-environment-configuration-env)
4. [User Guide: Streamlit Web Application](#4-user-guide-streamlit-web-application)
5. [User Guide: Interactive Terminal CLI](#5-user-guide-interactive-terminal-cli)
6. [Developer Guide: Running Master & Standalone OOP Pipeline Steps](#6-developer-guide-running-master--standalone-oop-pipeline-steps)
7. [Running Automated Unit Tests](#7-running-automated-unit-tests)
8. [Troubleshooting & FAQ](#8-troubleshooting--faq)

---

## 1. SYSTEM REQUIREMENTS

- **Operating System**: Windows 10/11, Linux, or macOS.
- **Python Version**: Python 3.9+ (Python 3.10 to 3.13 supported).
- **RAM**: Minimum 4 GB (8 GB recommended for local embedding processing).
- **Disk Space**: ~1 GB for virtual environment dependencies and local HuggingFace embedding weights.

---

## 2. INSTALLATION & SETUP

### Step 1: Open Terminal in Project Directory
Navigate to your Week 3 project root folder:
```bash
cd "d:\Sardar\Alphatron Technologies Research Internship\Batch 2\Week 3"
```

### Step 2: Create & Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. ENVIRONMENT CONFIGURATION (.env)

Create a file named **`.env`** in the root directory. Paste the following configuration:

```env
# Optional API Keys for online LLMs (e.g. Google Gemini)
GOOGLE_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Default Model Configuration
LLM_PROVIDER=gemini
LLM_MODEL_NAME=gemini-flash-latest

# Embeddings (Runs 100% locally for free using HuggingFace)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# ChromaDB Storage & Chunking Settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=rag_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
```

> **Note**: If `GOOGLE_API_KEY` is not provided, the chatbot automatically operates in **Offline Local Fallback Mode**, generating context summaries from retrieved documents without failing.

---

## 4. USER GUIDE: STREAMLIT WEB APPLICATION

To launch the modern glassmorphic web interface:

```bash
streamlit run app.py
```

### How to use the Web UI:
1. **Upload Documents**: Use the left sidebar file uploader to select **PDF, DOCX, TXT, or Excel/CSV** files.
2. **Process & Index**: Click **`⚡ Process & Index Documents`**. The app parses, chunks, and vectorizes your files into ChromaDB.
3. **Chat**: Type questions in the main chat input box at the bottom.
4. **Inspect Citations**: Expand **`📚 View Cited Sources`** under any answer to view exact document names, file format tags, page/sheet numbers, and text snippets.
5. **Clear Index**: Click **`🗑️ Clear DB`** in the sidebar to reset the ChromaDB database to 0 chunks whenever you want to start fresh with new documents.

---

## 5. USER GUIDE: INTERACTIVE TERMINAL CLI

To launch the interactive command-line chatbot:

```bash
python cli.py
```

### Supported Commands in CLI:
- `/ingest <file_or_directory>`: Ingest a document file or entire folder (e.g., `/ingest sample_data`).
- `/clear`: Reset and clear the ChromaDB vector collection.
- `/reset_memory`: Clear conversation chat history.
- `exit` or `quit`: Terminate the CLI chatbot session.

---

## 6. DEVELOPER GUIDE: RUNNING MASTER & STANDALONE OOP PIPELINE STEPS

### Option A: Run Full Master Pipeline (`Main.py`)
Executes all 5 modular OOP steps sequentially:
```bash
python Main.py
```

### Option B: Run Standalone Step Modules
Each step can be executed individually as an isolated OOP module:
```bash
python Step_1_DocumentIngestion.py   # Step 1: Ingestion & Chunking
python Step_2_EmbeddingFactory.py    # Step 2: Embedding Initialization
python Step_3_VectorStoreManager.py   # Step 3: ChromaDB Vector Store
python Step_4_RAGPipeline.py          # Step 4: Conversational RAG QA Chain
python Step_5_RAGEvaluator.py         # Step 5: Latency & Precision Evaluation
```

---

## 7. RUNNING AUTOMATED UNIT TESTS

Execute the comprehensive 9-test unit suite using `pytest`:

```bash
pytest -v
```

---

## 8. TROUBLESHOOTING & FAQ

### Q1: Streamlit shows "No readable text chunks found in uploaded documents"
- **Cause**: The PDF file may be a scanned image/photo without embedded digital text layers.
- **Solution**: Upload digital text PDFs, DOCX, TXT, or Excel/CSV spreadsheets.

### Q2: Why is the chatbot answering about old documents?
- **Cause**: ChromaDB is a persistent database and preserves previously indexed documents.
- **Solution**: Click **`🗑️ Clear DB`** in the Streamlit sidebar to wipe previous vectors before indexing a new file.

### Q3: How do I get a free Google Gemini API key?
- **Solution**: Visit [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey), click **Create API Key**, and paste it into your `.env` file (`GOOGLE_API_KEY=AIzaSy...`).
