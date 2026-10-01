"""Day-of-week-aware baselines and z-scores for daily KPIs.

For each day and metric, the baseline is built from the same weekday in
the previous `window_weeks` weeks (e.g. a Saturday is compared with the
eight prior Saturdays). Only *prior* observations are used, so the day
being scored can never contaminate its own baseline, and ordinary
weekday/weekend differences are not mistaken for anomalies.

Expects the output of clean_data(): one row per calendar day. A day gets
a baseline once at least `min_weeks` prior same-weekday values exist, using
up to `window_weeks` of them. The first `7 * min_weeks` days therefore have
no baseline (NaN), and days before `7 * window_weeks` use a shorter, noisier
baseline (reported in Baseline_N) instead of going unmonitored.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.data_loader import DATE_COLUMN, NUMERIC_COLUMNS

# 8 weeks beat 4 weeks on both precision and recall in the ground-truth
# sweep (scripts/evaluate_detection.py): a 4-sample baseline makes z-scores noisy.
WINDOW_WEEKS = 8
MIN_WEEKS = 4  # fewest prior weeks needed to score a day (warm-up fallback)


def compute_baselines(
    df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    window_weeks: int = WINDOW_WEEKS,
    min_weeks: int = MIN_WEEKS,
) -> pd.DataFrame:
    """Return one row per day x metric with baseline stats and a z-score.

    Columns: Date, Metric, Value, Baseline_Mean, Baseline_Std,
    Baseline_Q1, Baseline_Q3, Baseline_N, Z_Score. Baseline_N is the number
    of prior same-weekday weeks actually used (min_weeks..window_weeks).

    Baseline_Std is the sample standard deviation (ddof=1). Z_Score is NaN
    when there is no baseline or when the baseline std is 0.
    """
    if window_weeks < 2:
        raise ValueError("window_weeks must be at least 2 to compute a std")
    if min_weeks < 2:
        raise ValueError("min_weeks must be at least 2 to compute a std")
    min_weeks = min(min_weeks, window_weeks)

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
        n_prior = prior.notna().sum(axis=1)
        valid = n_prior >= min_weeks

        mean = prior.mean(axis=1).where(valid)
        std = prior.std(axis=1, ddof=1).where(valid)
        q1 = prior.quantile(0.25, axis=1).where(valid)
        q3 = prior.quantile(0.75, axis=1).where(valid)
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
                    "Baseline_N": n_prior.where(valid).to_numpy(),
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
