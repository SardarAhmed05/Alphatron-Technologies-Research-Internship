"""
Step 5: RAG System Evaluation & Latency Benchmarking (OOP)
Measures Retrieval Precision, Generation Latency, and Context Coverage
"""

from typing import List, Dict, Any, Optional
from src.evaluator import RAGEvaluator as CoreRAGEvaluator
from Step_3_VectorStoreManager import VectorStoreManager
from Step_4_RAGPipeline import ConversationalRAGPipeline


class RAGEvaluationPipeline:
    """
    Object-Oriented RAG Evaluator for benchmarking vector retrieval accuracy,
    measuring response latency (ms), and reporting citation quality.
    """

    def __init__(
        self,
        vector_manager: Optional[VectorStoreManager] = None,
        rag_pipeline: Optional[ConversationalRAGPipeline] = None,
    ):
        self.vector_manager = vector_manager or VectorStoreManager()
        self.rag_pipeline = rag_pipeline or ConversationalRAGPipeline(vector_manager=self.vector_manager)

        core_vector = self.vector_manager.core_manager if hasattr(self.vector_manager, "core_manager") else self.vector_manager
        core_rag = self.rag_pipeline.rag_chain if hasattr(self.rag_pipeline, "rag_chain") else self.rag_pipeline

        self.evaluator = CoreRAGEvaluator(vector_manager=core_vector, rag_pipeline=core_rag)

    def evaluate_query(self, query: str, expected_keyword: str = "") -> Dict[str, Any]:
        """Runs retrieval and end-to-end latency benchmarks for a single query."""
        print(f"[Step 5 - Evaluation] Benchmarking query: '{query}'...")
        retrieval_res = {}
        if expected_keyword:
            retrieval_res = self.evaluator.evaluate_retrieval(query, expected_keyword)
            print(f"[Step 5 - Evaluation] Retrieval Precision: {retrieval_res.get('precision', 0.0)*100:.1f}% ({retrieval_res.get('retrieval_latency_ms', 0)} ms)")

        e2e_res = self.evaluator.evaluate_end_to_end(query)
        print(f"[Step 5 - Evaluation] End-to-End Latency: {e2e_res.get('total_latency_ms', 0)} ms | Sources: {e2e_res.get('sources_count', 0)}")

        return {
            "retrieval": retrieval_res,
            "end_to_end": e2e_res
        }

    def run(self) -> Dict[str, Any]:
        """Executes Step 5 of the RAG pipeline."""
        print("\n" + "=" * 65)
        print("   STEP 5: RAG SYSTEM EVALUATION & LATENCY BENCHMARKING (OOP)")
        print("=" * 65)

        test_cases = [
            {"query": "What document formats are supported?", "expected_keyword": "PDF"},
            {"query": "What vector database is used?", "expected_keyword": "ChromaDB"},
        ]

        results = self.evaluator.run_full_evaluation(test_cases)
        print(f"[Step 5 - Evaluation] Evaluation complete across {results.get('total_evaluations', 0)} benchmark test cases.")
        print("[Step 5 - Evaluation] Step 5 completed successfully!\n")
        return results


if __name__ == "__main__":
    pipeline = RAGEvaluationPipeline()
    pipeline.run()
