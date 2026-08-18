import os
import tempfile
import streamlit as st
from src.loaders import DocumentIngestor
from src.vectorstore import VectorStoreManager
from src.rag_chain import ConversationalRAGChain, get_llm

# Page Configuration
st.set_page_config(
    page_title="Conversational RAG AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern Aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }
    .source-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        margin-top: 5px;
        font-size: 0.85rem;
    }
    .badge {
        background-color: #4f46e5;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_vector_manager():
    return VectorStoreManager()


def main():
    st.title("🤖 Multi-Format RAG AI Chatbot")
    st.caption("Powered by LangChain, ChromaDB & Multi-Format Document Ingestion (PDF, DOCX, TXT, Excel)")

    vector_manager = get_vector_manager()

    # Initialize RAG chain in session state
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = ConversationalRAGChain(vector_manager)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar: Document Upload & Vector DB Status
    with st.sidebar:
        st.header("📄 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, DOCX, TXT, Excel/CSV)",
            type=["pdf", "docx", "doc", "txt", "md", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
        )

        if st.button("Process & Index Documents", use_container_width=True):
            if uploaded_files:
                ingestor = DocumentIngestor()
                total_chunks = 0
                with st.spinner("Processing and indexing documents into ChromaDB..."):
                    temp_dir = tempfile.gettempdir()
                    for file in uploaded_files:
                        clean_name = os.path.basename(file.name)
                        tmp_path = os.path.join(temp_dir, f"upload_{clean_name}")

                        with open(tmp_path, "wb") as f:
                            f.write(file.getvalue())

                        try:
                            docs = ingestor.load_file(tmp_path)
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
                    st.success(f"Indexed {total_chunks} chunks successfully!")
                else:
                    st.warning("No readable text chunks found in uploaded documents.")
            else:
                st.warning("Please upload at least one document.")

        st.divider()

        # Database Stats & Controls
        st.header("📊 Vector DB Status")
        stats = vector_manager.get_collection_stats()
        st.metric("Total Indexed Chunks", stats.get("total_vector_chunks", 0))

        if st.button("Clear Vector Index", use_container_width=True):
            vector_manager.clear_database()
            st.session_state.rag_chain = ConversationalRAGChain(vector_manager)
            st.session_state.messages = []
            st.success("Database cleared!")
            st.rerun()

        if st.button("Reload LLM & Clear Memory", use_container_width=True):
            st.session_state.rag_chain = ConversationalRAGChain(vector_manager, llm=get_llm())
            st.session_state.messages = []
            st.info("LLM reloaded and chat memory reset!")
            st.rerun()

    # Main Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Cited Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"<span class='badge'>{src['file_type']}</span> **{src['file_name']}**",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Snippet: {src['snippet']}")

    user_query = st.chat_input("Ask a question about your uploaded documents...")
    if user_query:
        # Display human message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching ChromaDB & generating response..."):
                res = st.session_state.rag_chain.answer_question(user_query)
                answer = res["answer"]
                sources = res["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("📚 View Cited Sources"):
                        for src in sources:
                            st.markdown(
                                f"<span class='badge'>{src['file_type']}</span> **{src['file_name']}**",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Snippet: {src['snippet']}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
