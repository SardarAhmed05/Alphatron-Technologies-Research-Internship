import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

"""
=============================================================================
EDRIC - INTERACTIVE TERMINAL CLI
=============================================================================
Command-line interface for running autonomous web extraction and truth verification.
"""

import sys
import os
import json
from src.graph import EdricGraphManager
from src.exporter import DataExporter


def print_banner():
    print("""
=============================================================================
 🌐 EDRIC: Autonomous Web Intelligence & Truth Verification Engine (CLI)
=============================================================================
 Available Commands:
   /scrape <url>        : Scrape and extract structured intelligence from URL
   /search <topic>      : Execute live search and extract intelligence
   /export <csv|json>   : Export latest extracted records
   /history             : View session statistics
   exit / quit          : Exit CLI
=============================================================================
""")


def run_cli():
    print_banner()
    manager = EdricGraphManager(with_checkpointing=True)
    latest_state = None
    session_id = "cli-session-1"

    while True:
        try:
            user_input = input("\n[EDRIC-CLI] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting EDRIC CLI. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Exiting EDRIC CLI. Goodbye!")
            break

        elif user_input.startswith("/scrape"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /scrape <url>")
                continue
            url = parts[1].strip()
            print(f"\n[*] Initiating LangGraph Multi-Agent Scraping on: {url}")
            latest_state = manager.run(raw_input=url, input_type="url", thread_id=session_id)
            _display_summary(latest_state)

        elif user_input.startswith("/search"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /search <topic>")
                continue
            query = parts[1].strip()
            print(f"\n[*] Executing Live Topic Search & Verification on: '{query}'")
            latest_state = manager.run(raw_input=query, input_type="query", thread_id=session_id)
            _display_summary(latest_state)

        elif user_input.startswith("/export"):
            if not latest_state or not latest_state.get("dataframe_records"):
                print("No records available to export. Run /scrape or /search first.")
                continue
            parts = user_input.split(maxsplit=1)
            fmt = parts[1].strip().lower() if len(parts) > 1 else "csv"
            records = latest_state.get("dataframe_records", [])
            os.makedirs("./exports", exist_ok=True)
            
            if fmt == "json":
                path = "./exports/cli_export.json"
                DataExporter.to_json(records, filepath=path)
                print(f"[+] Exported JSON to: {path}")
            else:
                path = "./exports/cli_export.csv"
                DataExporter.to_csv(records, filepath=path)
                print(f"[+] Exported CSV to: {path}")

        elif user_input.startswith("/history") or user_input.startswith("/status"):
            if latest_state:
                print(f"Current Session Status:")
                print(f"- Last Input : {latest_state.get('raw_input')}")
                print(f"- Trust Score: {latest_state.get('trust_score')}%")
                print(f"- Records    : {len(latest_state.get('dataframe_records', []))}")
            else:
                print("No active extraction performed yet in this session.")

        else:
            print(f"Unknown command: '{user_input}'. Type '/scrape <url>' or '/search <topic>' or 'exit'.")


def _display_summary(state):
    records = state.get("dataframe_records", [])
    trust = state.get("trust_score", 0.0)
    meta = state.get("source_metadata", {})
    
    print("\n" + "=" * 60)
    print("📊 EXTRACTION & VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"• Domain / Source : {meta.get('domain', 'N/A')} ({meta.get('url', 'N/A')[:50]})")
    print(f"• Composite Trust : {trust}% (Threshold: 75%)")
    print(f"• Verification    : {'✓ PASS' if state.get('is_verified') else '⚠ PROVISIONAL'}")
    print(f"• Records Count   : {len(records)}")
    print("-" * 60)
    
    if records:
        print("\n[*] Sample Records (First 3):")
        df = DataExporter.to_dataframe(records[:3])
        print(df.to_string(index=False))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_cli()
