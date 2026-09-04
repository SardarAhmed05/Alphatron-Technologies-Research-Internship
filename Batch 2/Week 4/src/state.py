"""
EDRIC - LangGraph State Definition Module
Defines the typed state passed across all multi-agent nodes in the LangGraph StateGraph.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal


class SourceMetadata(TypedDict, total=False):
    url: str
    domain: str
    title: str
    status_code: int
    char_count: int
    fetch_time_ms: float
    is_live: bool
    headers: Dict[str, str]


class TrustBreakdown(TypedDict, total=False):
    domain_authority_score: float
    factual_consistency_score: float
    attribution_quality_score: float
    hallucination_risk_score: float
    overall_confidence: float
    flags: List[str]


class EdricState(TypedDict, total=False):
    # User Inputs
    input_type: Literal["url", "query", "text", "file"]
    raw_input: str
    extraction_goal: str
    custom_schema: Optional[str]

    # Web Ingestion & Cleaning
    cleaned_content: str
    source_metadata: SourceMetadata

    # Structured Extraction
    extracted_data: List[Dict[str, Any]]
    schema_fields: List[str]

    # Truth & Credibility Evaluation
    trust_score: float
    trust_breakdown: TrustBreakdown
    critique_feedback: str
    is_verified: bool
    iteration_count: int

    # Dual Outputs
    executive_briefing: str
    dataframe_records: List[Dict[str, Any]]

    # Execution & Traceability
    status_logs: List[str]
    current_node: str
    error_message: Optional[str]
