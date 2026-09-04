"""
EDRIC - Agent Node 2: Schema & Structured Entity Extractor
Transforms unstructured web Markdown into typed JSON schemas.
"""

import json
import re
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import EdricState
from src.config import get_llm


EXTRACTION_SYSTEM_PROMPT = """You are EDRIC's Neural Schema & Entity Extractor.
Your job is to read unstructured or semi-structured web content and extract high-value, structured data records as a clean JSON list of objects.

Rules:
1. Output MUST be valid JSON (a list of objects [ {"field1": "val1", ...}, ... ]).
2. Do NOT wrap in conversational text. Only return the JSON array or markdown JSON code block.
3. Preserve all numerical figures, currencies, dates, specifications, and factual metrics precisely.
4. If specific extraction goals are provided, strictly prioritize those fields.
5. If critique feedback is provided from a previous review, address those missing fields and corrections.
"""


def schema_extractor_node(state: EdricState) -> EdricState:
    """
    LangGraph Node: Extracts structured entity records from cleaned text.
    """
    content = state.get("cleaned_content", "")
    goal = state.get("extraction_goal") or "Extract all key entities, metrics, table data, and factual points."
    critique = state.get("critique_feedback", "")
    iteration = state.get("iteration_count", 0)
    status_logs = list(state.get("status_logs", []))

    status_logs.append(f"🧠 [Extractor] Extracting structured schema (Iteration {iteration + 1})...")

    # Construct prompt
    user_prompt = (
        f"EXTRACTION GOAL: {goal}\n\n"
        f"{f'CRITIQUE FEEDBACK TO FIX: {critique}' if critique else ''}\n\n"
        f"SOURCE CONTENT:\n{content[:25000]}"
    )

    extracted_records: List[Dict[str, Any]] = []

    try:
        llm = get_llm(temperature=0.1)
        response = llm.invoke([
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        
        if isinstance(response.content, str):
            raw_text = response.content.strip()
        elif isinstance(response.content, list) and response.content:
            first_item = response.content[0]
            raw_text = first_item.get("text", str(first_item)).strip() if isinstance(first_item, dict) else str(first_item).strip()
        else:
            raw_text = str(response.content).strip()
        
        # Strip markdown `json codeblock
        if "`" in raw_text:
            match = re.search(r"`(?:json)?\s*([\s\S]*?)\s*`", raw_text)
            if match:
                raw_text = match.group(1).strip()

        # Parse JSON
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            extracted_records = parsed
        elif isinstance(parsed, dict):
            # If wrapped in a top-level key like {"data": [...]} or single item
            for k, v in parsed.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    extracted_records = v
                    break
            if not extracted_records:
                extracted_records = [parsed]

    except Exception as e:
        status_logs.append(f"⚠️ [Extractor] LLM fallback engaged: {str(e)[:60]}")
        # Robust heuristic rule-based extraction fallback
        extracted_records = _heuristic_fallback_extractor(content, goal)

    # Clean schema fields - remove numeric fragments and malformed keys
    import re
    all_keys = set()
    for rec in extracted_records:
        if isinstance(rec, dict):
            for k in rec.keys():
                if isinstance(k, str) and not k.isdigit() and len(k) >= 2 and not re.match(r'^(?:\d+|\d{1,2}:\d{2})$', k):
                    all_keys.add(k)

    schema_fields = sorted(list(all_keys)) if all_keys else ["Verified Findings"]

    status_logs.append(
        f"✓ [Extractor] Successfully extracted {len(extracted_records)} structured records across {len(schema_fields)} schema fields."
    )

    return {
        **state,
        "extracted_data": extracted_records,
        "schema_fields": schema_fields,
        "status_logs": status_logs,
        "current_node": "extractor",
    }


def _heuristic_fallback_extractor(content: str, goal: str) -> List[Dict[str, Any]]:
    """Heuristic extractor parsing markdown tables, live blogs/timelines, news items, or bulleted points."""
    import re
    records = []
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    # 1. Check for Live Blog / Timelines (e.g. 10:54pm ..., Published 28 Aug, 2026 ...)
    timeline_matches = []
    for line in lines:
        clean_l = line.lstrip("-* #").strip()
        m = re.match(r'^(?:(?:Published|Updated)\s+[0-9A-Za-z,\s]+(?:\d{1,2}:\d{2}\s*(?:am|pm)?)|\d{1,2}:\d{2}\s*(?:am|pm)?)\s*(.*)', clean_l, re.I)
        if m:
            time_part = clean_l[:len(clean_l) - len(m.group(1))].strip(" :-")
            content_part = m.group(1).strip(" :-")
            if content_part and len(content_part) > 5:
                timeline_matches.append({
                    "Timestamp / Time": time_part or "Live Update",
                    "Headline / Update": content_part,
                })

    if len(timeline_matches) >= 2:
        return timeline_matches

    # 2. Check for Live News / Stories / Article / Index Blocks
    if any(prefix in content for prefix in ["### [Live News", "### [Story]", "### [Live Source]", "### [Web Result]", "### [Public Index Entry]", "### [Report Citation]"]):
        current_news: Dict[str, Any] = {}
        for line in lines:
            if line.startswith("### ["):
                if current_news and any(k in current_news for k in ["Headline", "Title", "Topic", "Resource / Finding"]):
                    records.append(current_news)
                headline = line.lstrip("#").strip()
                if "]" in headline:
                    headline = headline.split("]", 1)[1].strip()
                field_name = "Title" if "### [Story]" in content else "Resource / Finding" if "### [Public Index" in content else "Headline"
                current_news = {field_name: headline}
            elif line.startswith("- **Points**:") or line.startswith("- **Score**:"):
                current_news["Points / Score"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **Author**:") or line.startswith("- **By**:"):
                current_news["Author / Submitter"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **Published Date**:") or line.startswith("- **Published**:"):
                current_news["Published Date"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **Source Link**:") or line.startswith("- **Link**:"):
                current_news["Source Link"] = line.split(":", 1)[1].strip()
            elif line.startswith("- **Summary / Context**:") or line.startswith("- **Summary / Excerpt**:") or line.startswith("- **Summary**:"):
                current_news["Summary & Findings"] = line.split(":", 1)[1].strip()
            elif ":" in line and not line.startswith("|"):
                parts = line.split(":", 1)
                k = parts[0].lstrip("-* ").strip()
                if len(k) < 30 and parts[1].strip() and not k.isdigit():
                    current_news[k] = parts[1].strip()
        if current_news and any(k in current_news for k in ["Headline", "Title", "Topic", "Resource / Finding"]):
            records.append(current_news)

    # 3. Check for Markdown tables
    if not records:
        table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
        if len(table_lines) >= 3:
            headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
            for row_line in table_lines[2:]:
                cols = [c.strip() for c in row_line.strip("|").split("|")]
                if len(cols) == len(headers) and any(cols):
                    records.append(dict(zip(headers, cols)))

    # 4. Parse key-value headers & bullet points
    if not records:
        current_item: Dict[str, Any] = {}
        for line in lines:
            if line.startswith("#"):
                if current_item and len(current_item) > 1:
                    records.append(current_item)
                current_item = {"Topic / Section": line.lstrip("#").strip()}
            elif line.startswith("- ") or line.startswith("* "):
                bullet_text = line.lstrip("-* ").strip()
                if ":" in bullet_text:
                    parts = bullet_text.split(":", 1)
                    k, v = parts[0].strip(), parts[1].strip()
                    if len(k) < 35 and v and not k.isdigit():
                        current_item[k] = v
                elif len(bullet_text) > 10:
                    idx = len([k for k in current_item if k.startswith("Point")]) + 1
                    current_item[f"Point {idx}"] = bullet_text
            elif ":" in line and not line.startswith("|"):
                parts = line.split(":", 1)
                k, v = parts[0].strip(), parts[1].strip()
                if len(k) < 35 and v and not k.isdigit():
                    current_item[k] = v

        if current_item and len(current_item) > 1:
            records.append(current_item)

    # 5. Fallback: Parse paragraphs into individual findings
    if not records:
        paragraphs = [p for p in content.split("\n\n") if len(p.strip()) > 30]
        for i, p in enumerate(paragraphs[:8], 1):
            clean_p = p.replace("\n", " ").strip()
            records.append({
                "Item #": i,
                "Content Extract": clean_p[:200] + ("..." if len(clean_p) > 200 else ""),
                "Status": "Extracted"
            })

    if not records:
        records = [
            {"Title": "Extracted Document Summary", "Detail": content[:300], "Status": "Verified"}
        ]

    return records
