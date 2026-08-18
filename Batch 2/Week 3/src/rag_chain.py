from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

from src.config import (
    LLM_PROVIDER,
    LLM_MODEL_NAME,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    RETRIEVAL_K,
)
from src.prompts import QA_PROMPT, REPHRASE_QUESTION_PROMPT
from src.vectorstore import VectorStoreManager


class FallbackLocalLLM(BaseChatModel):
    """Fallback Runnable Chat Model when no external LLM API key is configured."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = (
            "[Local RAG Response]\nBased on the retrieved document context:\n"
            "I parsed your query. (Note: Set GOOGLE_API_KEY or OPENAI_API_KEY in .env to use a live online LLM)."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "fallback-local-llm"


def get_llm(provider: str = None, model_name: str = None):
    """Initialize and return the requested LLM instance."""
    provider = (provider or LLM_PROVIDER).lower()
    model_name = model_name or LLM_MODEL_NAME

    if provider == "gemini" and GOOGLE_API_KEY:
        # Candidates for active Google GenAI model endpoints
        candidates = [model_name, "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-pro-latest"]
        seen = []
        for cand in candidates:
            if cand and cand not in seen:
                seen.append(cand)

        for target_model in seen:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=target_model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                )
            except Exception as e:
                print(f"[Warning] Failed to load Gemini model '{target_model}': {e}")

    if provider == "openai" and OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name if "gpt" in model_name else "gpt-4o-mini",
                openai_api_key=OPENAI_API_KEY,
                temperature=0.2,
            )
        except Exception as e:
            print(f"[Warning] Failed to load OpenAI LLM ({e}). Falling back.")

    return FallbackLocalLLM()


class ConversationalRAGChain:
    """Manages conversational question-answering with ChromaDB vector context retrieval and memory."""

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        vector_manager: Optional[VectorStoreManager] = None,
        llm: Any = None,
        k: int = RETRIEVAL_K,
    ):
        self.vector_manager = vector_store_manager or vector_manager
        if not self.vector_manager:
            raise ValueError("Must provide vector_store_manager to ConversationalRAGChain.")
        self.llm = llm or get_llm()
        self.k = k
        self.chat_history: List[BaseMessage] = []

    def format_docs(self, docs: List[Document]) -> str:
        """Format retrieved document chunks into string context with metadata tags."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            file_name = doc.metadata.get("file_name", "Unknown Document")
            page = doc.metadata.get("page", "")
            sheet = doc.metadata.get("sheet_name", "")

            source_info = f"Document #{i}: {file_name}"
            if page != "":
                source_info += f" (Page {page + 1})"
            if sheet != "":
                source_info += f" (Sheet: {sheet})"

            formatted.append(f"[{source_info}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Execute RAG pipeline: reformulate question, retrieve vector context, format prompt, invoke LLM."""
        # Step 1: Rephrase question if chat history exists
        standalone_question = question
        if self.chat_history:
            try:
                rephrase_chain = REPHRASE_QUESTION_PROMPT | self.llm | StrOutputParser()
                standalone_question = rephrase_chain.invoke(
                    {"chat_history": self.chat_history, "input": question}
                )
            except Exception as e:
                print(f"[Warning] Question rephrasing failed: {e}")
                standalone_question = question

        # Step 2: Retrieve relevant vector chunks from ChromaDB
        docs = self.vector_manager.search(standalone_question, k=self.k)

        if not docs:
            answer = "I cannot find any relevant documents in the index to answer your question."
            sources = []
        else:
            formatted_context = self.format_docs(docs)

            # Step 3: Prompt engineering & LLM Generation with fallback handling
            try:
                qa_chain = QA_PROMPT | self.llm | StrOutputParser()
                answer = qa_chain.invoke(
                    {
                        "context": formatted_context,
                        "chat_history": self.chat_history,
                        "input": question,
                    }
                )
            except Exception as primary_err:
                print(f"[Warning] Primary LLM call failed ({primary_err}). Re-initializing LLM...")
                # Re-fetch LLM dynamically from current environment config
                self.llm = get_llm()
                try:
                    fallback_chain = QA_PROMPT | self.llm | StrOutputParser()
                    answer = fallback_chain.invoke(
                        {
                            "context": formatted_context,
                            "chat_history": self.chat_history,
                            "input": question,
                        }
                    )
                except Exception as fallback_err:
                    print(f"[Error] Fallback LLM call failed: {fallback_err}")
                    answer = (
                        f"[Document Context Answer]\nBased on retrieved context:\n\n"
                        f"{formatted_context[:500]}..."
                    )

            # Extract metadata for source citations
            sources = []
            for doc in docs:
                sources.append(
                    {
                        "file_name": doc.metadata.get("file_name", "Unknown"),
                        "file_type": doc.metadata.get("file_type", "Unknown"),
                        "page": doc.metadata.get("page", None),
                        "sheet_name": doc.metadata.get("sheet_name", None),
                        "snippet": doc.page_content[:150] + "...",
                    }
                )

        # Step 4: Update Conversation Memory
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs_count": len(docs),
            "standalone_question": standalone_question,
        }

    def clear_history(self):
        """Reset conversation memory."""
        self.chat_history = []
