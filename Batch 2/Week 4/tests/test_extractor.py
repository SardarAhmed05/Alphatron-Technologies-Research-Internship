"""
Unit tests for Extractor Node and heuristic fallback.
"""

from src.agents.extractor_node import schema_extractor_node, _heuristic_fallback_extractor
from src.state import EdricState


def test_heuristic_extractor_table():
    md_content = """
| Product | Price | Stock |
| --- | --- | --- |
| Laptop |  | 10 |
| Mouse |  | 50 |
"""
    records = _heuristic_fallback_extractor(md_content, "Extract products")
    assert len(records) == 2
    assert records[0]["Product"] == "Laptop"
    assert records[1]["Price"] == ""


def test_heuristic_extractor_key_value():
    text_content = """
# Company Profile
Company Name: QuantumTech
CEO: Dr. Alice Smith
Valuation: 
"""
    records = _heuristic_fallback_extractor(text_content, "Extract company profile")
    assert len(records) >= 1
    assert "QuantumTech" in str(records)
