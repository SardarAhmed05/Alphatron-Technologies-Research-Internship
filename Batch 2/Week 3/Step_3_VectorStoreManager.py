"""
Step 3: ChromaDB Vector Store & Persistent Indexing Manager (OOP)
Manages Document Vector Indexing, Similarity Search, and Database Persistence
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.vectorstore import VectorStoreManager as CoreVectorStoreManager
from src.config import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, RETRIEVAL_K


class VectorStoreManager:
    """
    Object-Oriented Vector Store Manager for persistent ChromaDB storage,
    document chunk indexing, similarity search, and collection stats.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Any = None,
    ):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.embedding_function = embedding_function
        self.core_manager = CoreVectorStoreManager(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, documents: List[Document]) -> int:
        """Splits raw documents into chunks and upserts them into ChromaDB."""
        if not documents:
            print("[Step 3 - VectorStore] No documents provided for indexing.")
            return 0

        print(f"[Step 3 - VectorStore] Splitting & indexing {len(documents)} raw documents into ChromaDB...")
        added_chunks = self.core_manager.add_documents(documents)
        print(f"[Step 3 - VectorStore] Indexed {added_chunks} vector chunks successfully.")
        return added_chunks

    def search(self, query: str, k: int = RETRIEVAL_K, search_type: str = "similarity") -> List[Document]:
        """Retrieves top-K relevant context document chunks for a query."""
        print(f"[Step 3 - VectorStore] Performing '{search_type}' search for query: '{query}' (k={k})...")
        results = self.core_manager.search(query, k=k, search_type=search_type)
        print(f"[Step 3 - VectorStore] Retrieved {len(results)} relevant vector chunks.")
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Returns statistics regarding collection count and storage paths."""
        stats = self.core_manager.get_collection_stats()
        print(f"[Step 3 - VectorStore] Collection Stats: Total Vectors = {stats.get('total_vector_chunks', 0)}")
        return stats

    def clear_database(self) -> bool:
        """Clears all document vectors from the ChromaDB collection."""
        print("[Step 3 - VectorStore] Resetting ChromaDB vector collection...")
        success = self.core_manager.clear_database()
        if success:
            print("[Step 3 - VectorStore] ChromaDB database cleared successfully.")
        return success

    def run(self, sample_docs: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Executes Step 3 of the RAG pipeline."""
        print("\n" + "=" * 65)
        print("   STEP 3: CHROMADB VECTOR STORE & INDEXING MANAGER (OOP)")
        print("=" * 65)
        if sample_docs:
            self.add_documents(sample_docs)
        stats = self.get_collection_stats()
        print("[Step 3 - VectorStore] Step 3 completed successfully!\n")
        return stats


if __name__ == "__main__":
    manager = VectorStoreManager()
    manager.run()
