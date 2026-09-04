"""
EDRIC - LangGraph StateGraph Construction & Execution Engine
Orchestrates the multi-agent state graph, conditional edges, reflection cycles, and memory checkpointing.
"""

from typing import Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import EdricState
from src.config import MAX_REFLECTION_CYCLES, MIN_CREDIBILITY_THRESHOLD
from src.agents.fetcher_node import web_fetcher_node
from src.agents.extractor_node import schema_extractor_node
from src.agents.verifier_node import legitimacy_verifier_node
from src.agents.synthesizer_node import intelligence_synthesizer_node


def route_after_verification(state: EdricState) -> Literal["extractor", "synthesizer"]:
    """
    Conditional Routing Edge:
    Evaluates whether the trust score meets criteria or requires reflection re-extraction.
    """
    trust_score = state.get("trust_score", 0.0)
    iteration = state.get("iteration_count", 0)
    is_verified = state.get("is_verified", False)

    if not is_verified and iteration < MAX_REFLECTION_CYCLES:
        return "extractor"  # Cyclic reflection loop
    return "synthesizer"   # Proceed to output generation


class EdricGraphManager:
    """
    Compiles, manages, and executes the LangGraph Multi-Agent Architecture for EDRIC.
    """

    def __init__(self, with_checkpointing: bool = True):
        self.with_checkpointing = with_checkpointing
        self.checkpointer = MemorySaver() if with_checkpointing else None
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the LangGraph StateGraph with nodes, edges, and cycles."""
        workflow = StateGraph(EdricState)

        # 1. Add Agent Nodes
        workflow.add_node("fetcher", web_fetcher_node)
        workflow.add_node("extractor", schema_extractor_node)
        workflow.add_node("verifier", legitimacy_verifier_node)
        workflow.add_node("synthesizer", intelligence_synthesizer_node)

        # 2. Add Deterministic Edges
        workflow.set_entry_point("fetcher")
        workflow.add_edge("fetcher", "extractor")
        workflow.add_edge("extractor", "verifier")

        # 3. Add Conditional Reflection Edge
        workflow.add_conditional_edges(
            "verifier",
            route_after_verification,
            {
                "extractor": "extractor",       # Self-Correction loop
                "synthesizer": "synthesizer",   # Proceed
            },
        )

        # 4. Terminate at Synthesizer
        workflow.add_edge("synthesizer", END)

        # Compile
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()

    def run(
        self,
        raw_input: str,
        input_type: str = "url",
        extraction_goal: str = "Extract all key structured metrics, tables, and entities.",
        thread_id: str = "edric-session-1",
    ) -> EdricState:
        """
        Executes the full LangGraph multi-agent pipeline for a given input.
        """
        initial_state: EdricState = {
            "input_type": input_type,  # type: ignore
            "raw_input": raw_input,
            "extraction_goal": extraction_goal,
            "iteration_count": 0,
            "status_logs": [f"🚀 [Engine] Initializing EDRIC LangGraph session ({thread_id})..."],
        }

        config = {"configurable": {"thread_id": thread_id}} if self.checkpointer else {}
        final_state = self.graph.invoke(initial_state, config=config)
        return final_state
