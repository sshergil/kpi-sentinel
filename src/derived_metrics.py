"""Ratio metrics that capture efficiency problems the raw metrics hide.

  Refund_Rate            = Refunds / Revenue
  Revenue_per_Ad_Dollar  = Revenue / Ad_Spend

A ratio is NaN when its denominator is zero or negative, rather than inf.
These columns go through the same baseline/detection pipeline as the raw
metrics. Because they are computed from other metrics, a flag on a ratio
can be a mechanical side effect (e.g. Refund_Rate rises when Revenue
collapses), so they are treated as supporting evidence, not as labelled
ground-truth metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import NUMERIC_COLUMNS

DERIVED_COLUMNS = ["Refund_Rate", "Revenue_per_Ad_Dollar"]
ALL_METRICS = list(NUMERIC_COLUMNS) + DERIVED_COLUMNS


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.astype(float)
    den = denominator.astype(float)
    return (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the derived ratio columns added."""
    missing = [c for c in ("Refunds", "Revenue", "Ad_Spend") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    out = df.copy()
    out["Refund_Rate"] = _safe_ratio(out["Refunds"], out["Revenue"])
    out["Revenue_per_Ad_Dollar"] = _safe_ratio(out["Revenue"], out["Ad_Spend"])
    return out
