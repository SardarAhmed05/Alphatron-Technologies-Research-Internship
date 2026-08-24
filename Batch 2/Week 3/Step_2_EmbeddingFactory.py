"""
Step 2: Vector Embedding Factory & Model Initialization (OOP)
Supports Local HuggingFace Sentence-Transformers, Google Gemini, and OpenAI Embeddings
"""

from typing import Any, Optional
from src.embeddings import get_embedding_function
from src.config import EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME


class EmbeddingModelFactory:
    """
    Object-Oriented Embedding Factory for instantiating, configuring,
    and verifying vector embedding generators across multiple providers.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = (provider or EMBEDDING_PROVIDER).lower()
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.embedding_function: Any = None

    def initialize_embeddings(self) -> Any:
        """Instantiates the specified embedding model function."""
        print(f"[Step 2 - Embeddings] Initializing provider='{self.provider}' model='{self.model_name}'...")
        self.embedding_function = get_embedding_function(
            provider=self.provider,
            model_name=self.model_name,
        )
        print("[Step 2 - Embeddings] Embedding model function initialized successfully.")
        return self.embedding_function

    def verify_embedding_dimension(self, sample_text: str = "RAG Test Query") -> int:
        """Generates a test vector embedding and returns its dimensionality."""
        if not self.embedding_function:
            self.initialize_embeddings()

        print(f"[Step 2 - Embeddings] Generating verification vector for sample query...")
        vector = self.embedding_function.embed_query(sample_text)
        dimension = len(vector)
        print(f"[Step 2 - Embeddings] Verification vector generated successfully (Dimensionality: {dimension}d).")
        return dimension

    def run(self) -> Any:
        """Executes Step 2 of the RAG pipeline."""
        print("\n" + "=" * 65)
        print("   STEP 2: VECTOR EMBEDDING FACTORY & INITIALIZATION (OOP)")
        print("=" * 65)
        emb_fn = self.initialize_embeddings()
        self.verify_embedding_dimension()
        print("[Step 2 - Embeddings] Step 2 completed successfully!\n")
        return emb_fn


if __name__ == "__main__":
    factory = EmbeddingModelFactory()
    factory.run()
