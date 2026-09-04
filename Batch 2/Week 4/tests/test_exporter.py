"""
Unit tests for Multi-Format Data Exporter.
"""

import os
import json
from src.exporter import DataExporter


def test_exporter_dataframe():
    records = [{"Name": "Alpha", "Score": 95}, {"Name": "Beta", "Score": 88}]
    df = DataExporter.to_dataframe(records)
    assert len(df) == 2
    assert list(df.columns) == ["Name", "Score"]


def test_exporter_csv():
    records = [{"ID": 1, "Item": "Widget"}]
    csv_str = DataExporter.to_csv(records)
    assert "ID,Item" in csv_str
    assert "1,Widget" in csv_str


def test_exporter_json():
    records = [{"Title": "Doc A", "Trust": 90}]
    json_str = DataExporter.to_json(records)
    parsed = json.loads(json_str)
    assert len(parsed) == 1
    assert parsed[0]["Title"] == "Doc A"
