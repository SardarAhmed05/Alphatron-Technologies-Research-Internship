"""
Master Conversational RAG Pipeline Orchestrator (OOP Standard)
Integrates and executes all 5 modular OOP steps sequentially.
"""

import sys
from Step_1_DocumentIngestion import DocumentIngestionPipeline
from Step_2_EmbeddingFactory import EmbeddingModelFactory
from Step_3_VectorStoreManager import VectorStoreManager
from Step_4_RAGPipeline import ConversationalRAGPipeline
from Step_5_RAGEvaluator import RAGEvaluationPipeline


class RAGBotMasterPipeline:
    """
    Master Object-Oriented Pipeline Orchestrator that connects all 5 modular steps:
      - Step 1: Multi-Format Document Ingestion & Chunking
      - Step 2: Vector Embedding Model Initialization
      - Step 3: ChromaDB Vector Indexing & Search
      - Step 4: Conversational RAG QA & Memory Chain
      - Step 5: System Evaluation & Latency Benchmarking
    """

    def __init__(self, sample_dir: str = "sample_data"):
        self.sample_dir = sample_dir
        self.step1_ingestion = DocumentIngestionPipeline(sample_dir=self.sample_dir)
        self.step2_embeddings = EmbeddingModelFactory()
        self.step3_vectorstore = VectorStoreManager()
        self.step4_rag = ConversationalRAGPipeline(vector_manager=self.step3_vectorstore)
        self.step5_evaluator = RAGEvaluationPipeline(
            vector_manager=self.step3_vectorstore,
            rag_pipeline=self.step4_rag
        )

    def run(self):
        """Executes all 5 pipeline steps sequentially."""
        print("\n" + "#" * 65)
        print("   MULTI-FORMAT CONVERSATIONAL RAG AI CHATBOT (5-STEP OOP PIPELINE)")
        print("#" * 65)

        # Step 1: Load and parse documents
        raw_docs = self.step1_ingestion.run()

        # Step 2: Initialize embedding factory
        self.step2_embeddings.run()

        # Step 3: Index documents into ChromaDB
        if raw_docs:
            self.step3_vectorstore.run(sample_docs=raw_docs)
        else:
            self.step3_vectorstore.run()

        # Step 4: Execute sample RAG chat QA interaction
        self.step4_rag.run(test_query="What document formats are supported by this system?")

        # Step 5: Execute system evaluation & latency benchmarks
        self.step5_evaluator.run()

        print("\n" + "#" * 65)
        print("   ALL 5 RAG PIPELINE STEPS EXECUTED AND COMPLETED SUCCESSFULLY!")
        print("#" * 65 + "\n")


if __name__ == "__main__":
    master_pipeline = RAGBotMasterPipeline()
    master_pipeline.run()
