"""Group flagged (day, metric) cells into incidents.

An incident is a run of consecutive days with at least one anomaly flag,
across all metrics. Related flags (e.g. conversion, orders and revenue
all dropping on the same day) therefore become one incident instead of
several unrelated alerts.

A sustained problem can dip back under the threshold for a day. Runs
separated by a short gap are therefore merged when they flag at least one
metric in common, so one sustained event is not reported as two alerts.
Runs with nothing in common stay separate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INCIDENT_COLUMNS = [
    "Incident_ID",
    "Start_Date",
    "End_Date",
    "Duration_Days",
    "Metrics",
    "Metric_Directions",
    "N_Flagged_Cells",
    "Peak_Abs_Z",
]


def build_incidents(
    detections: pd.DataFrame, max_gap_days: int = 1, bridge_gap_days: int = 2
) -> pd.DataFrame:
    """Return one row per incident from detect_anomalies() output.

    Flagged days at most `max_gap_days` apart always share an incident
    (1 = strictly consecutive days). Runs up to `bridge_gap_days` apart
    (2 = one unflagged day in between) also merge if they share a flagged
    metric; set `bridge_gap_days` equal to `max_gap_days` to disable that.
    `Metric_Directions` maps each flagged metric to "spike", "drop", or
    "mixed" when it moved both ways.
    """
    needed = ["Date", "Metric", "Z_Score", "Direction", "Is_Anomaly"]
    missing = [c for c in needed if c not in detections.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}; run detect_anomalies() first")

    flagged = detections.loc[detections["Is_Anomaly"], needed]
    if flagged.empty:
        return pd.DataFrame(columns=INCIDENT_COLUMNS)

    dates = pd.Series(sorted(flagged["Date"].unique()))
    starts_new_run = dates.diff().dt.days.fillna(np.inf) > max_gap_days
    run_by_date = dict(zip(dates, starts_new_run.cumsum()))
    flagged = flagged.assign(Run=flagged["Date"].map(run_by_date))

    # Merge neighbouring runs across a short gap when they share a metric.
    bridge_gap_days = max(bridge_gap_days, max_gap_days)
    incident_by_run: dict[int, int] = {}
    incident_id, group_metrics, previous_end = 0, set(), None
    for run_id, run in flagged.groupby("Run"):
        run_metrics = set(run["Metric"])
        bridge = (
            previous_end is not None
            and (run["Date"].min() - previous_end).days <= bridge_gap_days
            and bool(group_metrics & run_metrics)
        )
        if not bridge:
            incident_id += 1
            group_metrics = set()
        group_metrics |= run_metrics
        previous_end = run["Date"].max()
        incident_by_run[run_id] = incident_id
    flagged = flagged.assign(Incident_ID=flagged["Run"].map(incident_by_run))

    rows = []
    for incident_id, cells in flagged.groupby("Incident_ID"):
        directions = {
            metric: (g["Direction"].iloc[0] if g["Direction"].nunique() == 1 else "mixed")
            for metric, g in cells.groupby("Metric")
        }
        start, end = cells["Date"].min(), cells["Date"].max()
        rows.append(
            {
                "Incident_ID": int(incident_id),
                "Start_Date": start,
                "End_Date": end,
                "Duration_Days": (end - start).days + 1,
                "Metrics": sorted(directions),
                "Metric_Directions": directions,
                "N_Flagged_Cells": len(cells),
                "Peak_Abs_Z": float(cells["Z_Score"].abs().max()),
            }
        )
    return pd.DataFrame(rows, columns=INCIDENT_COLUMNS)
