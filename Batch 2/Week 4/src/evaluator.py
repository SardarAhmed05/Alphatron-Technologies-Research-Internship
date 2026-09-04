"""
EDRIC - System Evaluator & Performance Benchmark Suite
Evaluates extraction accuracy, reflection efficiency, latency (ms), and trust fidelity.
"""

import time
from typing import Dict, Any, List, Optional
from src.state import EdricState
from src.graph import EdricGraphManager


class EdricEvaluator:
    """
    Empirical evaluation suite benchmarking EDRIC's multi-agent scraping and verification pipeline.
    """

    def __init__(self, graph_manager: Optional[EdricGraphManager] = None):
        self.graph_manager = graph_manager or EdricGraphManager(with_checkpointing=False)

    def evaluate_pipeline(
        self,
        test_inputs: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Runs batch evaluation on test cases and computes statistical performance metrics.
        """
        results = []
        total_latency_ms = 0.0
        total_records = 0
        total_trust_score = 0.0
        successful_runs = 0
        reflections_triggered = 0

        for test in test_inputs:
            start_t = time.time()
            try:
                state = self.graph_manager.run(
                    raw_input=test["input"],
                    input_type=test.get("type", "url"),
                    extraction_goal=test.get("goal", "Extract key data"),
                )
                duration_ms = round((time.time() - start_t) * 1000, 2)
                total_latency_ms += duration_ms
                
                records = state.get("dataframe_records", [])
                trust = state.get("trust_score", 0.0)
                iterations = state.get("iteration_count", 1)

                if iterations > 1:
                    reflections_triggered += 1

                total_records += len(records)
                total_trust_score += trust
                successful_runs += 1

                results.append({
                    "input": test["input"][:40],
                    "status": "SUCCESS",
                    "records_count": len(records),
                    "trust_score": trust,
                    "latency_ms": duration_ms,
                    "reflection_cycles": iterations,
                })
            except Exception as err:
                duration_ms = round((time.time() - start_t) * 1000, 2)
                results.append({
                    "input": test["input"][:40],
                    "status": f"FAILED: {str(err)[:30]}",
                    "records_count": 0,
                    "trust_score": 0.0,
                    "latency_ms": duration_ms,
                    "reflection_cycles": 0,
                })

        count = max(1, len(test_inputs))
        avg_latency = round(total_latency_ms / count, 2)
        avg_trust = round(total_trust_score / max(1, successful_runs), 2)
        success_rate = round((successful_runs / count) * 100.0, 1)

        summary = {
            "total_test_cases": count,
            "success_rate_pct": success_rate,
            "avg_latency_ms": avg_latency,
            "avg_trust_score": avg_trust,
            "total_records_extracted": total_records,
            "reflections_triggered": reflections_triggered,
            "test_case_details": results,
        }

        return summary
