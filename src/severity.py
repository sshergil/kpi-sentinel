"""Heuristic severity score (0-100) for classified incidents.

Components (maximum points):
  magnitude  40  strongest |z| in the incident, capped at Z_CAP
  breadth    20  distinct RAW metrics flagged, capped at BREADTH_CAP
                 (derived ratios don't count: they move mechanically with
                 the raw metrics and would double-count the same event)
  duration   15  days affected, capped at DURATION_CAP
  revenue    25  revenue fell (25), moved both ways (15) or rose (10)

Incidents where only one raw metric is flagged are scaled down (x0.6)
unless that flag is extreme (|z| >= STRONG_Z): a weak lone flag with
nothing corroborating it is more likely noise. Positive-
impact incidents (e.g. a demand spike) are also scaled down because they
need review, not urgent action. Weights and label cut-offs are
transparent heuristics meant to be tuned, not statistically derived.
"""

from __future__ import annotations

import pandas as pd

from src.derived_metrics import DERIVED_COLUMNS

Z_CAP = 10.0
BREADTH_CAP = 4
DURATION_CAP = 5
POSITIVE_MULTIPLIER = 0.6
SINGLE_METRIC_MULTIPLIER = 0.6
STRONG_Z = 8.0  # a lone flag this extreme is trusted, not damped

REVENUE_POINTS = {"drop": 25.0, "mixed": 15.0, "spike": 10.0}

# (minimum score, label), checked from the top.
LABELS = [(75.0, "Critical"), (55.0, "High"), (35.0, "Medium"), (0.0, "Low")]


def _label(score: float) -> str:
    return next(label for cutoff, label in LABELS if score >= cutoff)


def _score_row(row: pd.Series) -> tuple[float, str]:
    magnitude = min(row["Peak_Abs_Z"], Z_CAP) / Z_CAP * 40
    n_raw = sum(m not in DERIVED_COLUMNS for m in row["Metrics"])
    breadth = min(n_raw, BREADTH_CAP) / BREADTH_CAP * 20
    duration = min(row["Duration_Days"], DURATION_CAP) / DURATION_CAP * 15
    revenue = REVENUE_POINTS.get(row["Metric_Directions"].get("Revenue"), 0.0)

    score = magnitude + breadth + duration + revenue
    note = ""
    if n_raw < 2 and row["Peak_Abs_Z"] < STRONG_Z:
        score *= SINGLE_METRIC_MULTIPLIER
        note += f" x{SINGLE_METRIC_MULTIPLIER} (weak single-metric flag)"
    if row.get("Impact") == "positive":
        score *= POSITIVE_MULTIPLIER
        note += f" x{POSITIVE_MULTIPLIER} (positive impact)"
    breakdown = (
        f"magnitude {magnitude:.1f} + breadth {breadth:.1f} + "
        f"duration {duration:.1f} + revenue {revenue:.1f}{note}"
    )
    return round(min(score, 100.0), 1), breakdown


def score_incidents(incidents: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Severity_Score, Severity_Label, Severity_Breakdown."""
    out = incidents.copy()
    if out.empty:
        for col in ("Severity_Score", "Severity_Label", "Severity_Breakdown"):
            out[col] = pd.Series(dtype="object")
        return out
    scored = [_score_row(row) for _, row in out.iterrows()]
    out["Severity_Score"] = [s for s, _ in scored]
    out["Severity_Label"] = [_label(s) for s, _ in scored]
    out["Severity_Breakdown"] = [b for _, b in scored]
    return out
