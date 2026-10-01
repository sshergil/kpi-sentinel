"""Measure how often Claude's incident summaries pass automatic fact-checking.

Usage (needs ANTHROPIC_API_KEY and `pip install anthropic`):
    python scripts/evaluate_summaries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.llm_eval import evaluate_summaries  # noqa: E402
from src.llm_summary import get_default_client  # noqa: E402
from src.service import analyze  # noqa: E402


def main() -> None:
    client = get_default_client()
    if client is None:
        sys.exit("Set ANTHROPIC_API_KEY (and `pip install anthropic`) to run this evaluation.")
    analysis = analyze(pd.read_csv(ROOT / "data" / "raw" / "ecommerce_daily_kpis.csv"))
    r = evaluate_summaries(analysis.incidents, analysis.detections, client)
    print(f"{r['llm_accepted']}/{r['n_incidents']} summaries passed fact-checking "
          f"(acceptance {r['acceptance_rate']:.0%}); {r['fallbacks']} fell back to the template.")
    if r["problem_types"]:
        print("Rejection reasons:", r["problem_types"])


if __name__ == "__main__":
    main()
