from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL_NAME,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
)


def get_embedding_function(provider: str = None, model_name: str = None) -> Any:
    """Initialize and return the requested embedding model handler."""
    provider = (provider or EMBEDDING_PROVIDER).lower()
    model_name = model_name or EMBEDDING_MODEL_NAME

    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)
    elif provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=GOOGLE_API_KEY
            )
        except Exception as e:
            print(f"[Warning] Failed to load Google Embeddings: {e}. Falling back to HuggingFace.")
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
            )
        except Exception as e:
            print(f"[Warning] Failed to load OpenAI Embeddings: {e}. Falling back to HuggingFace.")
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        # Fallback to HuggingFace
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
