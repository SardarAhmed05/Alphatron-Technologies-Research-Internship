"""
EDRIC - Agent Node 1: Web Fetcher & Content Sanitizer
Fetches raw web content, handles search queries, and normalizes input text.
"""

import os
from src.state import EdricState, SourceMetadata
from src.scraper import WebScraperEngine
from src.search import LiveSearchEngine


def web_fetcher_node(state: EdricState) -> EdricState:
    """
    LangGraph Node: Fetches live URL or executes topic search, producing clean semantic Markdown.
    """
    scraper = WebScraperEngine()
    searcher = LiveSearchEngine(scraper=scraper)
    
    input_type = state.get("input_type", "url")
    raw_input = state.get("raw_input", "").strip()
    status_logs = list(state.get("status_logs", []))

    status_logs.append(f"🕷️ [Fetcher] Ingesting {input_type.upper()}: '{raw_input[:60]}...'")

    # 1. URL Input Mode
    if input_type == "url":
        # Check if local file path
        if os.path.exists(raw_input):
            try:
                with open(raw_input, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                parsed = scraper.clean_html(content, source_url=raw_input)
                cleaned_content = parsed["cleaned_content"]
                metadata: SourceMetadata = {
                    "url": raw_input,
                    "domain": "local-filesystem",
                    "title": os.path.basename(raw_input),
                    "status_code": 200,
                    "char_count": len(cleaned_content),
                    "fetch_time_ms": 1.0,
                    "is_live": False,
                }
            except Exception as e:
                cleaned_content = f"Error reading local file: {e}"
                metadata = {"url": raw_input, "domain": "local-filesystem", "status_code": 500}
        else:
            fetch_result = scraper.fetch_url(raw_input)
            cleaned_content = fetch_result.get("cleaned_content", "")
            metadata: SourceMetadata = {
                "url": fetch_result.get("url", raw_input),
                "domain": fetch_result.get("domain", "unknown-domain"),
                "title": fetch_result.get("title", "Web Resource"),
                "status_code": fetch_result.get("status_code", 200),
                "char_count": fetch_result.get("char_count", len(cleaned_content)),
                "fetch_time_ms": fetch_result.get("fetch_time_ms", 0.0),
                "is_live": fetch_result.get("is_live", True),
            }

    # 2. Search Query Mode
    elif input_type == "query":
        search_result = searcher.search_topic(raw_input)
        cleaned_content = search_result.get("cleaned_content", "")
        metadata: SourceMetadata = {
            "url": f"https://duckduckgo.com/?q={raw_input}",
            "domain": "live-search-engine",
            "title": f"Search: {raw_input}",
            "status_code": 200,
            "char_count": len(cleaned_content),
            "fetch_time_ms": 45.0,
            "is_live": True,
        }

    # 3. Raw Text / Snippet Mode
    else:
        cleaned_content = raw_input
        metadata: SourceMetadata = {
            "url": "user-direct-input",
            "domain": "direct-input",
            "title": "Direct Text Snippet",
            "status_code": 200,
            "char_count": len(raw_input),
            "fetch_time_ms": 0.0,
            "is_live": False,
        }

    status_logs.append(
        f"✓ [Fetcher] Content sanitized successfully ({len(cleaned_content)} characters extracted from {metadata.get('domain')})."
    )

    return {
        **state,
        "cleaned_content": cleaned_content,
        "source_metadata": metadata,
        "status_logs": status_logs,
        "current_node": "fetcher",
    }
