import os
import shutil
import pytest
from langchain_core.documents import Document
from Step_3_VectorStoreManager import VectorStoreManager
from Step_4_RAGPipeline import ConversationalRAGPipeline
from Step_5_RAGEvaluator import RAGEvaluationPipeline
from src.rag_chain import FallbackLocalLLM


@pytest.fixture
def test_eval_setup(tmp_path):
    persist_dir = str(tmp_path / "test_eval_chroma")
    vector_mgr = VectorStoreManager(persist_dir=persist_dir, collection_name="test_eval")

    sample_docs = [
        Document(
            page_content="ChromaDB is a vector database supporting fast vector similarity search.",
            metadata={"file_name": "vector_info.txt", "file_type": "TXT", "page": 0},
        )
    ]
    vector_mgr.add_documents(sample_docs)

    rag_pipe = ConversationalRAGPipeline(vector_manager=vector_mgr, llm=FallbackLocalLLM())
    eval_pipe = RAGEvaluationPipeline(vector_manager=vector_mgr, rag_pipeline=rag_pipe)

    yield eval_pipe

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_evaluator_metrics(test_eval_setup):
    res = test_eval_setup.evaluate_query("What vector database is used?", expected_keyword="ChromaDB")
    assert "retrieval" in res
    assert "end_to_end" in res
    assert res["retrieval"]["precision"] == 1.0
    assert res["end_to_end"]["sources_count"] > 0
