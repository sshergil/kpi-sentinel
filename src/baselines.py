"""Day-of-week-aware baselines and z-scores for daily KPIs.

For each day and metric, the baseline is built from the same weekday in
the previous `window_weeks` weeks (e.g. a Saturday is compared with the
four prior Saturdays). Only *prior* observations are used, so the day
being scored can never contaminate its own baseline, and ordinary
weekday/weekend differences are not mistaken for anomalies.

Expects the output of clean_data(): one row per calendar day. A day gets
a baseline only when all `window_weeks` prior same-weekday values exist,
so the first `7 * window_weeks` days have no baseline (NaN).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.data_loader import DATE_COLUMN, NUMERIC_COLUMNS

WINDOW_WEEKS = 4


def compute_baselines(
    df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    window_weeks: int = WINDOW_WEEKS,
) -> pd.DataFrame:
    """Return one row per day x metric with baseline stats and a z-score.

    Columns: Date, Metric, Value, Baseline_Mean, Baseline_Std,
    Baseline_Q1, Baseline_Q3, Z_Score.

    Baseline_Std is the sample standard deviation (ddof=1). Z_Score is NaN
    when there is no baseline or when the baseline std is 0.
    """
    if window_weeks < 2:
        raise ValueError("window_weeks must be at least 2 to compute a std")

    metrics = list(metrics) if metrics is not None else list(NUMERIC_COLUMNS)
    missing = [c for c in [DATE_COLUMN] + metrics if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    data = df.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    _require_daily_calendar(data.index)

    frames = []
    for metric in metrics:
        s = data[metric].astype(float)

        # Columns are the same weekday 1..window_weeks weeks earlier.
        prior = pd.concat([s.shift(7 * k) for k in range(1, window_weeks + 1)], axis=1)
        complete = prior.notna().all(axis=1)

        mean = prior.mean(axis=1).where(complete)
        std = prior.std(axis=1, ddof=1).where(complete)
        q1 = prior.quantile(0.25, axis=1).where(complete)
        q3 = prior.quantile(0.75, axis=1).where(complete)
        z = ((s - mean) / std).where(std > 0)

        frames.append(
            pd.DataFrame(
                {
                    DATE_COLUMN: s.index,
                    "Metric": metric,
                    "Value": s.to_numpy(),
                    "Baseline_Mean": mean.to_numpy(),
                    "Baseline_Std": std.to_numpy(),
                    "Baseline_Q1": q1.to_numpy(),
                    "Baseline_Q3": q3.to_numpy(),
                    "Z_Score": z.to_numpy(),
                }
            )
        )

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)


def _require_daily_calendar(index: pd.Index) -> None:
    """Raise if dates are not one consecutive row per day."""
    steps = pd.Series(index).diff().dropna()
    if not (steps == pd.Timedelta(days=1)).all():
        raise ValueError(
            "Dates must be one consecutive row per day; run clean_data() first"
        )
