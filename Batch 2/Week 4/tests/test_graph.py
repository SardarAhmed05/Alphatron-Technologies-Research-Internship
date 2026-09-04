"""
Unit tests for LangGraph StateGraph Construction and Edge Routing.
"""

from src.graph import EdricGraphManager, route_after_verification
from src.state import EdricState


def test_graph_manager_init():
    manager = EdricGraphManager(with_checkpointing=False)
    assert manager.graph is not None


def test_route_after_verification():
    # Pass condition
    state_pass: EdricState = {
        "trust_score": 88.0,
        "is_verified": True,
        "iteration_count": 1,
    }
    assert route_after_verification(state_pass) == "synthesizer"

    # Fail & retry condition
    state_retry: EdricState = {
        "trust_score": 50.0,
        "is_verified": False,
        "iteration_count": 1,
    }
    assert route_after_verification(state_retry) == "extractor"

    # Max cycles reached
    state_max: EdricState = {
        "trust_score": 50.0,
        "is_verified": False,
        "iteration_count": 3,
    }
    assert route_after_verification(state_max) == "synthesizer"
