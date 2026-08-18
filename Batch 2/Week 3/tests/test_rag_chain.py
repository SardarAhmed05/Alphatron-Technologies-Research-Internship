import os
import shutil
import pytest
from langchain_core.documents import Document
from src.vectorstore import VectorStoreManager
from src.rag_chain import ConversationalRAGChain, FallbackLocalLLM


@pytest.fixture
def test_rag_setup(tmp_path):
    persist_dir = str(tmp_path / "test_rag_chroma")
    manager = VectorStoreManager(persist_dir=persist_dir, collection_name="test_rag")

    sample_docs = [
        Document(
            page_content="The Advanced RAG System is led by Lead Architect Alice Smith with a target latency of < 500ms.",
            metadata={"file_name": "project_info.txt", "file_type": "TXT", "page": 0},
        )
    ]
    manager.add_documents(sample_docs)

    # Use FallbackLocalLLM for unit tests to ensure fast offline execution
    rag_chain = ConversationalRAGChain(vector_manager=manager, llm=FallbackLocalLLM())
    yield rag_chain

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_answer_generation(test_rag_setup):
    res = test_rag_setup.answer_question("Who leads the Advanced RAG System?")
    assert "answer" in res
    assert "sources" in res
    assert len(res["sources"]) > 0
    assert res["sources"][0]["file_name"] == "project_info.txt"


def test_conversational_history(test_rag_setup):
    test_rag_setup.answer_question("What is the system latency target?")
    assert len(test_rag_setup.chat_history) == 2

    test_rag_setup.clear_history()
    assert len(test_rag_setup.chat_history) == 0
