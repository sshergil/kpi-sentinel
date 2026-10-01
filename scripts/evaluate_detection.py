"""Sweep baseline window and z-threshold, scoring each against ground truth.

Usage (from the project root):
    python scripts/evaluate_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.anomaly_detection import detect_anomalies  # noqa: E402
from src.baselines import compute_baselines  # noqa: E402
from src.data_cleaning import clean_data  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.evaluation import expand_labels, load_ground_truth, score_detections  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "ecommerce_daily_kpis.csv"
TRUTH_PATH = ROOT / "data" / "raw" / "ground_truth_anomalies.csv"

WINDOWS = (4, 8)
Z_THRESHOLDS = (3.0, 4.0, 5.0)


def main() -> None:
    raw, _ = load_data(DATA_PATH)
    cleaned, _ = clean_data(raw)
    truth = load_ground_truth(TRUTH_PATH)
    labels = expand_labels(truth)

    results = []
    details = {}
    for window in WINDOWS:
        baselines = compute_baselines(cleaned, window_weeks=window)
        for z in Z_THRESHOLDS:
            summary, per_event = score_detections(
                detect_anomalies(baselines, z_threshold=z), labels, window
            )
            results.append({"window_weeks": window, "z_threshold": z, **summary})
            details[(window, z)] = per_event

    table = pd.DataFrame(results)
    shown = table.assign(
        events=table["events_detected"].astype(str) + "/" + table["events_scorable"].astype(str)
    )[["window_weeks", "z_threshold", "precision", "recall", "f1", "tp", "fp", "fn", "events", "echo_fp", "cells_unscorable"]]
    print(f"{len(labels)} labelled (day, metric) cells across {len(truth)} events\n")
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    best = table.sort_values(["f1", "recall"], ascending=False).iloc[0]
    key = (int(best["window_weeks"]), float(best["z_threshold"]))
    print(f"\nPer-event detail for best F1 (window={key[0]}, z>{key[1]:g}):")
    print(details[key].to_string(index=False))


if __name__ == "__main__":
    main()
