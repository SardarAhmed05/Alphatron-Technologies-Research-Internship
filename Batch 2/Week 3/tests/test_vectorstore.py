import os
import shutil
import pytest
from langchain_core.documents import Document
from src.vectorstore import VectorStoreManager


@pytest.fixture
def temp_vector_store(tmp_path):
    persist_dir = str(tmp_path / "test_chroma")
    manager = VectorStoreManager(persist_dir=persist_dir, collection_name="test_collection")
    yield manager
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_chunking_and_indexing(temp_vector_store):
    sample_docs = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. "
            "It enables retrieval-augmented generation (RAG) by connecting LLMs to external data sources.",
            metadata={"file_name": "test_doc.txt", "file_type": "TXT"},
        )
    ]

    added_count = temp_vector_store.add_documents(sample_docs)
    assert added_count > 0

    stats = temp_vector_store.get_collection_stats()
    assert stats["total_vector_chunks"] == added_count


def test_similarity_search(temp_vector_store):
    sample_docs = [
        Document(
            page_content="ChromaDB is an open-source AI-native vector database designed for fast similarity search.",
            metadata={"file_name": "chroma.txt", "file_type": "TXT"},
        ),
        Document(
            page_content="Python is a high-level programming language widely used in AI and data science.",
            metadata={"file_name": "python.txt", "file_type": "TXT"},
        ),
    ]
    temp_vector_store.add_documents(sample_docs)

    results = temp_vector_store.search("What is ChromaDB?", k=1)
    assert len(results) == 1
    assert "ChromaDB" in results[0].page_content
