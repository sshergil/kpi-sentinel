"""Run the full detection pipeline on the real data and inspect the incidents.

Usage (from the project root):
    python scripts/run_pipeline.py

Writes data/processed/incidents.csv and prints the highest-severity
incidents plus an event-level comparison with the ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data_cleaning import clean_data  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.evaluation import load_ground_truth, score_incidents  # noqa: E402
from src.pipeline import run_detection_pipeline  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "ecommerce_daily_kpis.csv"
TRUTH_PATH = ROOT / "data" / "raw" / "ground_truth_anomalies.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "incidents.csv"


def main() -> None:
    raw, _ = load_data(DATA_PATH)
    cleaned, _ = clean_data(raw)
    _, incidents = run_detection_pipeline(cleaned)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    incidents.assign(
        Metrics=incidents["Metrics"].map(", ".join),
        Metric_Directions=incidents["Metric_Directions"].map(str),
    ).to_csv(OUTPUT_PATH, index=False)

    print(f"{len(incidents)} incidents found; saved to {OUTPUT_PATH.relative_to(ROOT)}\n")
    show = incidents.sort_values("Severity_Score", ascending=False).head(20)
    print(
        show[["Incident_ID", "Start_Date", "End_Date", "Duration_Days", "Category",
              "Severity_Score", "Severity_Label", "Evidence"]]
        .assign(Start_Date=show["Start_Date"].dt.date, End_Date=show["End_Date"].dt.date)
        .to_string(index=False, max_colwidth=60)
    )

    truth = load_ground_truth(TRUTH_PATH)
    summary, per_event, matched = score_incidents(incidents, truth)
    print("\nEvent-level comparison with ground truth (1-day tolerance):")
    print(f"  events detected:    {summary['events_detected']}/{summary['n_events']}")
    print(f"  incidents matching an event: {summary['incidents_matched']}/{summary['n_incidents']}"
          f" (precision {summary['incident_precision']:.2f})")
    if len(incidents):
        print(f"  mean severity, matching incidents:     {incidents.loc[matched, 'Severity_Score'].mean():.1f}")
        print(f"  mean severity, non-matching incidents: {incidents.loc[~matched, 'Severity_Score'].mean():.1f}")
    print()
    print(
        per_event.assign(Start_Date=per_event["Start_Date"].dt.date, End_Date=per_event["End_Date"].dt.date)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
