import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.embeddings import get_embedding_function


class VectorStoreManager:
    """Manages document chunking, ChromaDB vector indexing, persistence, and retrieval."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Any = None,
    ):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.embedding_function = embedding_function or get_embedding_function()

        os.makedirs(self.persist_dir, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_dir,
        )

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> List[Document]:
        """Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
        if not documents:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        # Assign chunk index metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        return chunks

    def add_documents(self, documents: List[Document]) -> int:
        """Process and index documents into ChromaDB."""
        if not documents:
            return 0
        chunks = self.split_documents(documents)
        if not chunks:
            return 0
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def search(
        self, query: str, k: int = 4, search_type: str = "similarity"
    ) -> List[Document]:
        """Retrieve relevant context documents for a query."""
        if search_type == "mmr":
            return self.vector_store.max_marginal_relevance_search(query, k=k)
        return self.vector_store.similarity_search(query, k=k)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics regarding indexed documents and vector count."""
        try:
            collection = self.vector_store._collection
            return {
                "total_vector_chunks": collection.count(),
                "persist_directory": self.persist_dir,
                "collection_name": self.collection_name,
            }
        except Exception as e:
            return {"total_vector_chunks": 0, "error": str(e)}

    def clear_database(self) -> bool:
        """Clear all indexed documents from the ChromaDB collection."""
        try:
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_dir,
            )
            return True
        except Exception as e:
            print(f"[Error] Failed to reset Chroma database: {e}")
            return False
