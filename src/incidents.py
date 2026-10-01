"""Group flagged (day, metric) cells into incidents.

An incident is a run of consecutive days with at least one anomaly flag,
across all metrics. Related flags (e.g. conversion, orders and revenue
all dropping on the same day) therefore become one incident instead of
several unrelated alerts.
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


def build_incidents(detections: pd.DataFrame, max_gap_days: int = 1) -> pd.DataFrame:
    """Return one row per incident from detect_anomalies() output.

    Flagged days at most `max_gap_days` apart belong to the same incident
    (1 = strictly consecutive days). `Metric_Directions` maps each flagged
    metric to "spike", "drop", or "mixed" when it moved both ways.
    """
    needed = ["Date", "Metric", "Z_Score", "Direction", "Is_Anomaly"]
    missing = [c for c in needed if c not in detections.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}; run detect_anomalies() first")

    flagged = detections.loc[detections["Is_Anomaly"], needed]
    if flagged.empty:
        return pd.DataFrame(columns=INCIDENT_COLUMNS)

    dates = pd.Series(sorted(flagged["Date"].unique()))
    starts_new = dates.diff().dt.days.fillna(np.inf) > max_gap_days
    id_by_date = dict(zip(dates, starts_new.cumsum()))
    flagged = flagged.assign(Incident_ID=flagged["Date"].map(id_by_date))

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
