"""Score detected anomalies against the known ground-truth anomalies.

Scoring is done at the (day, metric) level: a ground-truth event that
affects Revenue and Orders for 3 days contributes 6 labelled cells, and a
detection counts as correct only if it flags one of those exact cells.

Cells with no baseline (the warm-up period) are excluded from scoring so
a longer baseline window is not penalised for days it cannot score.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import NUMERIC_COLUMNS


def load_ground_truth(path) -> pd.DataFrame:
    """Read ground_truth_anomalies.csv with parsed start/end dates."""
    return pd.read_csv(path, parse_dates=["start_date", "end_date"])


def expand_labels(ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Expand events into one row per (Event_ID, Date, Metric)."""
    rows = []
    for event_id, event in enumerate(ground_truth.itertuples(index=False), start=1):
        metrics = [m.strip() for m in str(event.metrics_affected).split(",")]
        unknown = [m for m in metrics if m not in NUMERIC_COLUMNS]
        if unknown:
            raise ValueError(f"Event {event_id}: unknown metric(s) {', '.join(unknown)}")
        for date in pd.date_range(event.start_date, event.end_date):
            for metric in metrics:
                rows.append({"Event_ID": event_id, "Date": date, "Metric": metric})
    return pd.DataFrame(rows, columns=["Event_ID", "Date", "Metric"])


def score_detections(
    detections: pd.DataFrame, labels: pd.DataFrame, window_weeks: int
) -> tuple[dict, pd.DataFrame]:
    """Compare detect_anomalies() output with expanded ground-truth labels.

    Returns (summary, per_event). `echo_fp` counts false positives that sit
    exactly 1..window_weeks weeks after a labelled cell of the same metric:
    an earlier anomaly is inside their baseline, so these are likely
    side effects of baseline contamination rather than random noise.
    """
    scored = detections.loc[detections["Baseline_Mean"].notna()]
    scorable = set(zip(scored["Date"], scored["Metric"]))
    flagged = scored.loc[scored["Is_Anomaly"]]
    predicted = set(zip(flagged["Date"], flagged["Metric"]))
    label_keys = set(zip(labels["Date"], labels["Metric"]))

    tp = predicted & label_keys
    fp = predicted - label_keys
    fn = (label_keys & scorable) - predicted

    echo_fp = {
        (d, m)
        for d, m in fp
        if any((d - pd.Timedelta(weeks=k), m) in label_keys for k in range(1, window_weeks + 1))
    }

    per_event = labels.assign(
        Scorable=[k in scorable for k in zip(labels["Date"], labels["Metric"])],
        Detected=[k in predicted for k in zip(labels["Date"], labels["Metric"])],
    )
    per_event = (
        per_event.groupby("Event_ID")
        .agg(Cells_Labeled=("Metric", "size"), Cells_Scorable=("Scorable", "sum"), Cells_Detected=("Detected", "sum"))
        .reset_index()
    )

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    summary = {
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "events_detected": int((per_event["Cells_Detected"] > 0).sum()),
        "events_scorable": int((per_event["Cells_Scorable"] > 0).sum()),
        "echo_fp": len(echo_fp),
        "cells_unscorable": len(label_keys - scorable),
    }
    return summary, per_event
