# EDRIC Technical Architecture Documentation

## 1. System Overview
EDRIC (Engine for Dynamic Research, Intelligence and Credibility) is an autonomous multi-agent intelligence extraction and truth verification system built using LangGraph, LangChain, and LLMs (Google Gemini / OpenAI).

## 2. Core State Schema (src/state.py)
State is tracked through the EdricState TypedDict:
- input_type: URL, Query, Text, or File.
- raw_input: Raw URL, query, or text provided by user.
- extraction_goal: Natural language specification of target entities.
- cleaned_content: Sanitized semantic Markdown string.
- source_metadata: Metadata dictionary (domain, status_code, char_count, fetch_time_ms, is_live).
- extracted_data: List of structured JSON objects.
- schema_fields: List of identified column/field keys.
- trust_score: Composite legitimacy score (0.0 to 100.0).
- trust_breakdown: Detailed sub-scores for domain authority, factuality, completeness, and hallucination risk.
- critique_feedback: Actionable critique string for reflection cycles.
- iteration_count: Current reflection loop count.
- executive_briefing: Rendered Markdown executive briefing.
- dataframe_records: Normalized tabular records.
- status_logs: Chronological execution trace.

## 3. LangGraph Workflow and Edge Routing (src/graph.py)

### Node Definitions:
1. fetcher: Ingests URL/query/text, prunes DOM, computes Domain Trust Index.
2. extractor: Prompts LLM for zero-shot JSON schema extraction with heuristic fallback.
3. verifier: Audits factuality against source text, scores hallucination risk, computes composite trust.
4. synthesizer: Normalizes records into DataFrames and drafts executive briefing.

## 4. Domain Trust Index (DTI) Formulation
The Domain Trust Index evaluates:
- Authoritative Top-Level Domains (.gov, .edu, .org, .io, .ai, .com)
- Known High-Reputation Whitelist Catalog (e.g., arxiv.org, nature.com, reuters.com, github.com, wikipedia.org)
- Composite Formula: Trust Score = 0.35 * DTI + 0.35 * FactualScore + 0.20 * Completeness + 0.10 * (100 - HallucinationRisk)