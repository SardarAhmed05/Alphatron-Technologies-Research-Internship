"""
Step 4: Conversational RAG Pipeline & QA Memory Chain (OOP)
Handles Conversation Memory, Standalone Question Rephrasing, System Prompts, and Citation Formatting
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.rag_chain import ConversationalRAGChain as CoreConversationalRAGChain, FallbackLocalLLM
from Step_3_VectorStoreManager import VectorStoreManager


class ConversationalRAGPipeline:
    """
    Object-Oriented Conversational RAG Pipeline that manages context retrieval,
    standalone question rephrasing, system prompt formatting, and LLM question-answering.
    """

    def __init__(
        self,
        vector_manager: Optional[VectorStoreManager] = None,
        llm: Any = None,
    ):
        self.vector_manager = vector_manager or VectorStoreManager()
        self.core_manager = self.vector_manager.core_manager if hasattr(self.vector_manager, "core_manager") else self.vector_manager
        self.llm = llm
        self.rag_chain = CoreConversationalRAGChain(
            vector_manager=self.core_manager,
            llm=self.llm,
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Executes full RAG retrieval and LLM answer generation for a user query."""
        print(f"[Step 4 - RAG Pipeline] Processing user query: '{question}'...")
        result = self.rag_chain.answer_question(question)
        print(f"[Step 4 - RAG Pipeline] Response generated successfully ({len(result.get('sources', []))} sources cited).")
        return result

    def clear_history(self) -> None:
        """Resets conversation chat history."""
        print("[Step 4 - RAG Pipeline] Clearing conversation memory...")
        self.rag_chain.clear_history()
        print("[Step 4 - RAG Pipeline] Conversation memory cleared.")

    def run(self, test_query: str = "Summarize the uploaded documents.") -> Dict[str, Any]:
        """Executes Step 4 of the RAG pipeline."""
        print("\n" + "=" * 65)
        print("   STEP 4: CONVERSATIONAL RAG PIPELINE & MEMORY CHAIN (OOP)")
        print("=" * 65)
        result = self.answer_question(test_query)
        print(f"\n[Step 4 - Answer Preview]:\n{result.get('answer', '')}\n")
        print("[Step 4 - RAG Pipeline] Step 4 completed successfully!\n")
        return result


if __name__ == "__main__":
    pipeline = ConversationalRAGPipeline()
    pipeline.run()
