import sys
import argparse
from src.loaders import DocumentIngestor
from src.vectorstore import VectorStoreManager
from src.rag_chain import ConversationalRAGChain


def main():
    parser = argparse.ArgumentParser(description="Multi-Format RAG AI Chatbot CLI")
    parser.add_argument(
        "--ingest",
        type=str,
        help="Path to a document file or directory to ingest into ChromaDB",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a single question to the chatbot",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the vector database index",
    )
    args = parser.parse_args()

    vector_manager = VectorStoreManager()

    if args.clear:
        vector_manager.clear_database()
        print("Successfully cleared ChromaDB vector index.")
        return

    if args.ingest:
        ingestor = DocumentIngestor()
        print(f"Ingesting documents from: {args.ingest}")
        import os

        if os.path.isdir(args.ingest):
            docs = ingestor.load_directory(args.ingest)
        else:
            docs = ingestor.load_file(args.ingest)

        count = vector_manager.add_documents(docs)
        print(f"Successfully indexed {count} text chunks into ChromaDB!")

    if args.query:
        rag_chain = ConversationalRAGChain(vector_manager)
        result = rag_chain.answer_question(args.query)
        print("\n=== Answer ===")
        print(result["answer"])
        print("\n=== Sources ===")
        for s in result["sources"]:
            print(f"- {s['file_name']} (Type: {s['file_type']})")


if __name__ == "__main__":
    main()
