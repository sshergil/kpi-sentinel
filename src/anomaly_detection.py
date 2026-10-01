"""Flag anomalies in day x metric baseline tables.

Two independent statistical checks are applied to the output of
compute_baselines():

  * Z-score:  |z| > z_threshold
  * IQR rule: value outside [Q1 - k*IQR, Q3 + k*IQR] of the prior
              same-weekday window

A day is reported as an anomaly only when BOTH checks agree. Z_Flag and
IQR_Flag are kept as separate columns so each check can be evaluated on
its own.

Defaults (8-week window, |z| > 4) were chosen from a sweep against the
project's ground truth (scripts/evaluate_detection.py): F1 0.77, versus
0.36-0.52 for a 4-week window. The IQR agreement rule is close to
redundant with the z-score (a large |z| almost always falls outside the
IQR fences too), so window length and z threshold are what drive results.

This module only detects. Severity, business rules and LLM explanations
come later and never decide what is anomalous.
"""

from __future__ import annotations

import pandas as pd

Z_THRESHOLD = 4.0
IQR_MULTIPLIER = 1.5


def detect_anomalies(
    baselines: pd.DataFrame,
    z_threshold: float = Z_THRESHOLD,
    iqr_multiplier: float = IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Return a copy of `baselines` with detection columns added.

    Added columns: IQR_Lower, IQR_Upper, Z_Flag, IQR_Flag, Is_Anomaly,
    Direction ("spike" or "drop" for anomalies, missing otherwise).

    Rows without a baseline (NaN) are never flagged.
    """
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")
    if iqr_multiplier < 0:
        raise ValueError("iqr_multiplier must be non-negative")

    required = ["Value", "Baseline_Mean", "Baseline_Q1", "Baseline_Q3", "Z_Score"]
    missing = [c for c in required if c not in baselines.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}; run compute_baselines() first")

    out = baselines.copy()

    iqr = out["Baseline_Q3"] - out["Baseline_Q1"]
    out["IQR_Lower"] = out["Baseline_Q1"] - iqr_multiplier * iqr
    out["IQR_Upper"] = out["Baseline_Q3"] + iqr_multiplier * iqr

    # Comparisons with NaN are False, so rows without a baseline stay unflagged.
    out["Z_Flag"] = out["Z_Score"].abs() > z_threshold
    out["IQR_Flag"] = (out["Value"] < out["IQR_Lower"]) | (out["Value"] > out["IQR_Upper"])
    out["Is_Anomaly"] = out["Z_Flag"] & out["IQR_Flag"]

    direction = pd.Series("drop", index=out.index).where(out["Value"] <= out["Baseline_Mean"], "spike")
    out["Direction"] = direction.where(out["Is_Anomaly"])
    return out
