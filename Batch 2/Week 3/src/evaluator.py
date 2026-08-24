import time
from typing import List, Dict, Any
from langchain_core.documents import Document


class RAGEvaluator:
    """
    Object-Oriented Evaluation Module for assessing RAG retrieval precision,
    response generation latency, and context overlap.
    """

    def __init__(self, vector_manager: Any = None, rag_pipeline: Any = None):
        self.vector_manager = vector_manager
        self.rag_pipeline = rag_pipeline

    def evaluate_retrieval(self, query: str, expected_keyword: str, k: int = 4) -> Dict[str, Any]:
        """Evaluates vector retrieval precision and keyword presence in retrieved context."""
        if not self.vector_manager:
            raise ValueError("VectorStoreManager must be provided for retrieval evaluation.")

        start_time = time.time()
        retrieved_docs: List[Document] = self.vector_manager.search(query, k=k)
        retrieval_latency_ms = (time.time() - start_time) * 1000

        hits = sum(1 for doc in retrieved_docs if expected_keyword.lower() in doc.page_content.lower())
        precision = hits / len(retrieved_docs) if retrieved_docs else 0.0

        return {
            "query": query,
            "expected_keyword": expected_keyword,
            "retrieved_count": len(retrieved_docs),
            "keyword_hits": hits,
            "precision": round(precision, 4),
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        }

    def evaluate_end_to_end(self, query: str) -> Dict[str, Any]:
        """Evaluates total end-to-end RAG pipeline latency and output attributes."""
        if not self.rag_pipeline:
            raise ValueError("ConversationalRAGPipeline must be provided for end-to-end evaluation.")

        start_time = time.time()
        result = self.rag_pipeline.answer_question(query)
        total_latency_ms = (time.time() - start_time) * 1000

        return {
            "query": query,
            "answer_length": len(result.get("answer", "")),
            "sources_count": len(result.get("sources", [])),
            "total_latency_ms": round(total_latency_ms, 2),
        }

    def run_full_evaluation(self, test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Executes a batch evaluation over a set of test queries."""
        eval_results = []
        for test_case in test_queries:
            q = test_case["query"]
            keyword = test_case.get("expected_keyword", "")

            retrieval_stats = {}
            if self.vector_manager and keyword:
                retrieval_stats = self.evaluate_retrieval(q, keyword)

            e2e_stats = {}
            if self.rag_pipeline:
                e2e_stats = self.evaluate_end_to_end(q)

            eval_results.append({
                "query": q,
                "retrieval": retrieval_stats,
                "end_to_end": e2e_stats
            })

        return {
            "total_evaluations": len(eval_results),
            "results": eval_results
        }
