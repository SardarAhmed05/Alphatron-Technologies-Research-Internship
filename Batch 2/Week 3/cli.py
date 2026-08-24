import os
from Step_1_DocumentIngestion import DocumentIngestionPipeline
from Step_3_VectorStoreManager import VectorStoreManager
from Step_4_RAGPipeline import ConversationalRAGPipeline


def run_cli():
    print("=" * 60)
    print("   🤖 Multi-Format Conversational RAG AI Chatbot (OOP)   ")
    print("=" * 60)

    vector_manager = VectorStoreManager()
    stats = vector_manager.get_collection_stats()
    print(f"Indexed Chunks in Database: {stats.get('total_vector_chunks', 0)}")
    print("Commands:")
    print("  /ingest <path>  - Ingest a document file or folder")
    print("  /clear          - Clear the vector database index")
    print("  /reset_memory   - Clear conversation history")
    print("  exit / quit     - Exit chatbot")
    print("-" * 60)

    rag_pipeline = ConversationalRAGPipeline(vector_manager=vector_manager)

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if user_input.startswith("/ingest"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: /ingest <file_or_directory_path>")
                    continue
                path = parts[1].strip()
                if not os.path.exists(path):
                    print(f"Error: Path '{path}' does not exist.")
                    continue
                ingest_pipe = DocumentIngestionPipeline()
                if os.path.isdir(path):
                    docs = ingest_pipe.load_directory(path)
                else:
                    docs = ingest_pipe.load_single_file(path)
                added = vector_manager.add_documents(docs)
                print(f"Successfully indexed {added} chunks into ChromaDB!")
                continue

            if user_input == "/clear":
                vector_manager.clear_database()
                print("ChromaDB vector store cleared.")
                continue

            if user_input == "/reset_memory":
                rag_pipeline.clear_history()
                print("Conversation history reset.")
                continue

            # Process RAG Question
            response = rag_pipeline.answer_question(user_input)
            print(f"\nAI > {response['answer']}")

            if response["sources"]:
                print("\n📌 Cited Sources:")
                for idx, src in enumerate(response["sources"], 1):
                    info = f"  [{idx}] {src['file_name']} ({src['file_type']})"
                    if src.get("page") is not None:
                        info += f" - Page {src['page'] + 1}"
                    if src.get("sheet_name"):
                        info += f" - Sheet: {src['sheet_name']}"
                    print(info)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    run_cli()
