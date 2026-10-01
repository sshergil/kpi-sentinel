"""Print plain-English summaries for the highest-severity incidents.

Usage (from the project root):
    python scripts/summarize_incidents.py [N]

Uses the Anthropic API if ANTHROPIC_API_KEY is set (and `anthropic` is
installed); otherwise prints the deterministic template summaries.
Every LLM answer is verified against the incident facts before display.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_cleaning import clean_data  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.llm_summary import get_default_client, summarize_incidents  # noqa: E402
from src.pipeline import run_detection_pipeline  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "ecommerce_daily_kpis.csv"


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    raw, _ = load_data(DATA_PATH)
    cleaned, _ = clean_data(raw)
    detections, incidents = run_detection_pipeline(cleaned)
    top = incidents.sort_values("Severity_Score", ascending=False).head(top_n)

    client = get_default_client()
    print(f"Summaries via: {'Anthropic API' if client else 'template (no API key set)'}\n")
    for _, row in summarize_incidents(top, detections, client).iterrows():
        print(f"[{row['Severity_Label']} {row['Severity_Score']:g}] {row['Category']} "
              f"(source: {row['Summary_Source']})")
        print(f"  {row['Summary']}")
        for check in row["Checks"]:
            print(f"  - Check: {check}")
        print()


if __name__ == "__main__":
    main()
