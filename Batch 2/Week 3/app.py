import os
import tempfile
import streamlit as st
from Step_1_DocumentIngestion import DocumentIngestionPipeline
from Step_3_VectorStoreManager import VectorStoreManager
from Step_4_RAGPipeline import ConversationalRAGPipeline
from src.rag_chain import get_llm
from src.config import GOOGLE_API_KEY, OPENAI_API_KEY, LLM_PROVIDER, LLM_MODEL_NAME

# Page Configuration
st.set_page_config(
    page_title="Conversational RAG AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Glassmorphic CSS Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Header Gradient */
    .gradient-header {
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #e879f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.25rem;
        margin-bottom: 0.25rem;
    }

    /* Status Badges */
    .status-badge-online {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-badge-offline {
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid rgba(234, 179, 8, 0.4);
        color: #facc15;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    .format-tag {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6);
        background: linear-gradient(90deg, #4338ca 0%, #6d28d9 100%);
    }

    /* Source Citation Cards */
    .source-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.85rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_vector_manager():
    return VectorStoreManager()


def main():
    # Hero Header Section
    col_header, col_status = st.columns([3, 1])
    with col_header:
        st.markdown("<h1 class='gradient-header'>🤖 Conversational RAG AI Assistant</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div>"
            "<span class='format-tag'>PDF</span>"
            "<span class='format-tag'>DOCX</span>"
            "<span class='format-tag'>TXT</span>"
            "<span class='format-tag'>EXCEL / CSV</span>"
            "<span class='format-tag'>ChromaDB Vector Store</span>"
            "<span class='format-tag'>LangChain OOP Pipeline</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Determine LLM connection status
    is_online = bool((LLM_PROVIDER == "gemini" and GOOGLE_API_KEY) or (LLM_PROVIDER == "openai" and OPENAI_API_KEY))
    with col_status:
        st.write("")
        if is_online:
            st.markdown(
                f"<div style='text-align: right;'><span class='status-badge-online'>🟢 Online LLM ({LLM_MODEL_NAME})</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align: right;'><span class='status-badge-offline'>🟡 Offline Local Fallback</span></div>",
                unsafe_allow_html=True,
            )

    st.write("")

    vector_manager = get_vector_manager()

    # Initialize session state objects
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = ConversationalRAGPipeline(vector_manager=vector_manager)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar: Document Ingestion & Database Management
    with st.sidebar:
        st.markdown("### 📄 Document Ingestion")
        st.caption("Upload multi-format files to index into ChromaDB.")
        
        uploaded_files = st.file_uploader(
            "Select Documents",
            type=["pdf", "docx", "doc", "txt", "md", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
            help="Upload PDF, Word, Plain Text, or Excel/CSV spreadsheets.",
        )

        if st.button("⚡ Process & Index Documents", use_container_width=True):
            if uploaded_files:
                ingestion_pipe = DocumentIngestionPipeline()
                total_chunks = 0
                with st.spinner("Parsing, chunking, and vectorizing documents..."):
                    temp_dir = tempfile.gettempdir()
                    for file in uploaded_files:
                        clean_name = os.path.basename(file.name)
                        tmp_path = os.path.join(temp_dir, f"upload_{clean_name}")

                        with open(tmp_path, "wb") as f:
                            f.write(file.getvalue())

                        try:
                            docs = ingestion_pipe.load_single_file(tmp_path)
                            for doc in docs:
                                doc.metadata["file_name"] = file.name
                            added = vector_manager.add_documents(docs)
                            total_chunks += added
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")
                        finally:
                            if os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass

                if total_chunks > 0:
                    st.success(f"Successfully indexed {total_chunks} chunks into ChromaDB!")
                    st.rerun()
                else:
                    st.warning("No readable text chunks found in uploaded documents.")
            else:
                st.warning("Please select at least one document file.")

        st.divider()

        # Database Status Metrics
        st.markdown("### 📊 Vector DB Status")
        stats = vector_manager.get_collection_stats()
        total_vectors = stats.get("total_vector_chunks", 0)
        
        st.metric("Total Indexed Vector Chunks", f"{total_vectors:,}")
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            if st.button("🗑️ Clear DB", use_container_width=True):
                vector_manager.clear_database()
                st.session_state.rag_pipeline = ConversationalRAGPipeline(vector_manager=vector_manager)
                st.session_state.messages = []
                st.success("Vector DB cleared!")
                st.rerun()

        with col_db2:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.rag_pipeline = ConversationalRAGPipeline(vector_manager=vector_manager, llm=get_llm())
                st.session_state.messages = []
                st.info("Chat history cleared!")
                st.rerun()

    # Interactive Sample Prompts (Displayed when chat is empty)
    if not st.session_state.messages:
        st.markdown("### 💡 Quick Prompt Starters")
        st.caption("Click any starter query below or type your custom question:")
        col_p1, col_p2, col_p3 = st.columns(3)
        
        starter_query = None
        with col_p1:
            if st.button("📄 Summarize Documents", use_container_width=True):
                starter_query = "Summarize the key information contained in all uploaded documents."
        with col_p2:
            if st.button("📊 List Key Metrics & Tables", use_container_width=True):
                starter_query = "What data tables, metrics, or employee information are present in the documents?"
        with col_p3:
            if st.button("⚙️ Show Technical Specifications", use_container_width=True):
                starter_query = "What system architecture, guidelines, or operational targets are specified?"

        if starter_query:
            st.session_state.messages.append({"role": "user", "content": starter_query})
            st.rerun()

    # Chat Messages Display Loop
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Cited Sources"):
                    for idx, src in enumerate(msg["sources"], 1):
                        file_type = src.get("file_type", "DOC")
                        file_name = src.get("file_name", "Unknown File")
                        page_info = f" (Page {src['page'] + 1})" if src.get("page") is not None else ""
                        sheet_info = f" (Sheet: {src['sheet_name']})" if src.get("sheet_name") else ""
                        
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<strong>[{idx}] {file_name}</strong> <span class='format-tag'>{file_type}</span>{page_info}{sheet_info}<br>"
                            f"<em style='color: #94a3b8;'>\"{src.get('snippet', '')}\"</em>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # Chat Input Box
    user_query = st.chat_input("Ask any question about your uploaded documents...")
    if user_query:
        # Append and display user query
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching ChromaDB vector store & generating response..."):
                res = st.session_state.rag_pipeline.answer_question(user_query)
                answer = res.get("answer", "No answer generated.")
                sources = res.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📚 View Cited Sources"):
                        for idx, src in enumerate(sources, 1):
                            file_type = src.get("file_type", "DOC")
                            file_name = src.get("file_name", "Unknown File")
                            page_info = f" (Page {src['page'] + 1})" if src.get("page") is not None else ""
                            sheet_info = f" (Sheet: {src['sheet_name']})" if src.get("sheet_name") else ""
                            
                            st.markdown(
                                f"<div class='source-box'>"
                                f"<strong>[{idx}] {file_name}</strong> <span class='format-tag'>{file_type}</span>{page_info}{sheet_info}<br>"
                                f"<em style='color: #94a3b8;'>\"{src.get('snippet', '')}\"</em>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
