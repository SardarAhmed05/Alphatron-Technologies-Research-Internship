# RAG AI Chatbot - Architecture Diagram

The system architecture consists of 5 modular layers:
1. **User Interface Layer**: Streamlit Web UI and CLI Terminal Interface.
2. **Document Ingestion Layer**: Multi-format loaders (PDF, DOCX, TXT, Excel/CSV).
3. **Indexing & Vector DB Layer**: Recursive Character Chunking, Embedding Models (HuggingFace/Google/OpenAI), and persistent ChromaDB storage.
4. **Retrieval & Memory Layer**: Conversational Query Rephrasing, Top-K Similarity / MMR search, and Chat Memory.
5. **Generation & LLM Layer**: System Prompt Engineering and LLM Provider (Google Gemini / OpenAI / Local).

```mermaid
graph TD
    subgraph UI ["1. User Interface Layer"]
        A1[Streamlit Web App app.py]
        A2[CLI Terminal Interface cli.py]
    end

    subgraph Ingestion ["2. Document Ingestion Layer"]
        B1[PDF Loader PyPDF]
        B2[DOCX Loader Docx2txt]
        B3[TXT Loader TextLoader]
        B4[Excel / CSV Loader Pandas]
    end

    subgraph VectorDB ["3. Indexing & Vector DB Layer"]
        C1[Text Splitter RecursiveCharacter]
        C2[Embedding Generator HuggingFace / Gemini]
        C3[(ChromaDB Vector Store ./chroma_db)]
    end

    subgraph Retrieval ["4. Retrieval & Memory Layer"]
        D1[Question Rephraser Standalone Query]
        D2[Vector Similarity / MMR Retriever]
        D3[Chat History Memory]
    end

    subgraph Generation ["5. Generation & LLM Layer"]
        E1[RAG System Prompt Context Grounding]
        E2[LLM Engine Google Gemini / OpenAI]
        E3[Formatted Answer + Cited Sources]
    end

    A1 -->|Upload Documents| Ingestion
    A2 -->|Upload Documents| Ingestion
    Ingestion --> C1
    C1 --> C2
    C2 --> C3

    A1 -->|User Query| D1
    A2 -->|User Query| D1
    D3 <--> D1
    D1 -->|Standalone Query| D2
    C3 <-->|K Nearest Vectors| D2
    D2 -->|Context Chunks| E1
    E1 --> E2
    E2 --> E3
    E3 --> A1
    E3 --> A2
```
