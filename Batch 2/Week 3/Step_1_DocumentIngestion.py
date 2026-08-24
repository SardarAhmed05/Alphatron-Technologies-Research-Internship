"""
Step 1: Multi-Format Document Ingestion & Chunking Pipeline (OOP)
Supports PDF, DOCX, TXT/MD, and Excel/CSV Spreadsheet Parsing
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from src.loaders import DocumentIngestor
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentIngestionPipeline:
    """
    Modular Object-Oriented Document Ingestion Pipeline for multi-format document loading,
    metadata enrichment, and RecursiveCharacterTextSplitter document chunking.
    """

    def __init__(
        self,
        sample_dir: str = "sample_data",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.sample_dir = sample_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ingestor = DocumentIngestor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.loaded_documents: List[Document] = []

    def load_single_file(self, file_path: str) -> List[Document]:
        """Loads a single document file with format-specific parsing."""
        print(f"[Step 1 - Ingestion] Loading single document: {file_path}...")
        docs = self.ingestor.load_file(file_path)
        print(f"[Step 1 - Ingestion] Loaded {len(docs)} page/sheet documents from {Path(file_path).name}.")
        return docs

    def load_directory(self, directory_path: Optional[str] = None) -> List[Document]:
        """Recursively scans and loads all supported documents from a directory."""
        target_dir = directory_path or self.sample_dir
        print(f"[Step 1 - Ingestion] Scanning directory for multi-format documents: {target_dir}...")
        self.loaded_documents = self.ingestor.load_directory(target_dir)
        print(f"[Step 1 - Ingestion] Total raw documents loaded: {len(self.loaded_documents)}")
        return self.loaded_documents

    def run(self) -> List[Document]:
        """Executes Step 1 of the RAG pipeline."""
        print("\n" + "=" * 65)
        print("   STEP 1: MULTI-FORMAT DOCUMENT INGESTION & PARSING (OOP)")
        print("=" * 65)
        if os.path.exists(self.sample_dir):
            docs = self.load_directory(self.sample_dir)
        else:
            print(f"[Step 1 - Ingestion] Directory '{self.sample_dir}' not found. Initializing empty dataset.")
            docs = []

        print("[Step 1 - Ingestion] Step 1 completed successfully!\n")
        return docs


if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    pipeline.run()
