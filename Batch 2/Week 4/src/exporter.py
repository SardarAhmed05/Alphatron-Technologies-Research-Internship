"""
EDRIC - Multi-Format Data Exporter
Converts extracted records into Pandas DataFrames, CSV, Excel (XLSX), JSON, and Markdown tables.
"""

import json
from typing import List, Dict, Any, Optional
import pandas as pd


class DataExporter:
    """
    Serializes structured intelligence into industry-standard tabular and machine-readable formats.
    """

    @staticmethod
    def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts raw list of dictionary records to a clean Pandas DataFrame."""
        if not records:
            return pd.DataFrame(columns=["Status", "Message"], data=[["Empty", "No records extracted"]])
        df = pd.DataFrame(records)
        return df

    @classmethod
    def to_csv(cls, records: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
        """Exports records to CSV format string or writes to disk."""
        df = cls.to_dataframe(records)
        csv_data = df.to_csv(index=False)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(csv_data)
        return csv_data

    @classmethod
    def to_excel(cls, records: List[Dict[str, Any]], filepath: str) -> bool:
        """Writes records to an Excel .xlsx workbook."""
        df = cls.to_dataframe(records)
        try:
            df.to_excel(filepath, index=False, engine="openpyxl")
            return True
        except Exception:
            return False

    @classmethod
    def to_json(cls, records: List[Dict[str, Any]], filepath: Optional[str] = None, indent: int = 2) -> str:
        """Exports records to formatted JSON string or writes to disk."""
        json_data = json.dumps(records, indent=indent, default=str)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_data)
        return json_data

    @classmethod
    def to_markdown(cls, records: List[Dict[str, Any]]) -> str:
        """Converts records into a GitHub-flavored Markdown table."""
        df = cls.to_dataframe(records)
        return df.to_markdown(index=False)
