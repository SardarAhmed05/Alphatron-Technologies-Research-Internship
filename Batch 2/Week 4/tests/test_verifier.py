"""
Unit tests for Verifier Node and Trust Scoring.
"""

from src.agents.verifier_node import legitimacy_verifier_node
from src.state import EdricState


def test_legitimacy_verifier_logic():
    state: EdricState = {
        "input_type": "url",
        "raw_input": "https://arxiv.org/abs/2401.001",
        "source_metadata": {"url": "https://arxiv.org/abs/2401.001", "domain": "arxiv.org"},
        "cleaned_content": "This paper presents a new optimization algorithm with 99% accuracy.",
        "extracted_data": [{"Metric": "Accuracy", "Value": "99%"}],
        "iteration_count": 0,
        "status_logs": [],
    }

    verified_state = legitimacy_verifier_node(state)
    assert verified_state["trust_score"] >= 75.0
    assert verified_state["is_verified"] is True
    assert "domain_authority_score" in verified_state["trust_breakdown"]
