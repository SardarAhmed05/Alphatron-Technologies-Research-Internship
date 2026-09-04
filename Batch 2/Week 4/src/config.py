"""
EDRIC - Configuration & Environment Module
Loads environment variables, manages API keys, and initializes LLM instances.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv(override=True)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM Engine Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-flash-latest")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Web Scraping Configuration
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "35000"))
DEFAULT_USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)

# LangGraph Guardrails
MAX_REFLECTION_CYCLES = int(os.getenv("MAX_REFLECTION_CYCLES", "2"))
MIN_CREDIBILITY_THRESHOLD = int(os.getenv("MIN_CREDIBILITY_THRESHOLD", "75"))


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = TEMPERATURE,
) -> BaseChatModel:
    """
    Factory function returning an initialized LangChain ChatModel instance.
    Supports Google Gemini, OpenAI, and local mock fallback.
    """
    provider = (provider or LLM_PROVIDER).lower()
    
    # Priority: Gemini
    if provider == "gemini" or (GOOGLE_API_KEY and provider != "openai"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = model_name or LLM_MODEL_NAME or "gemini-1.5-flash"
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=GOOGLE_API_KEY,
                max_retries=1,
                timeout=12,
            )
        except Exception as e:
            # Fallback to OpenAI if configured
            if OPENAI_API_KEY:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model_name or "gpt-4o-mini",
                    temperature=temperature,
                    api_key=OPENAI_API_KEY,
                )
            raise RuntimeError(f"Failed to initialize Gemini LLM: {e}")

    # OpenAI Provider
    elif provider == "openai" or OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )

    raise ValueError(
        "No valid LLM API key detected. Please configure GOOGLE_API_KEY or OPENAI_API_KEY in your .env file."
    )
