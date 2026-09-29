"""Clean raw KPI data.

Fixes data-quality problems only: duplicates, missing days, missing or
non-numeric cells, and impossible negative values.

It deliberately does NOT touch values that are unusual but real (extreme
spikes/drops, genuine zero-value days). Deciding whether those are
anomalies is the detector's job, not cleaning's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data_loader import DATE_COLUMN, NUMERIC_COLUMNS, REQUIRED_COLUMNS, SchemaError


@dataclass
class CleaningReport:
    """Counts of what cleaning changed."""

    rows_in: int = 0
    rows_out: int = 0
    unparseable_date_rows_dropped: int = 0
    duplicate_rows_dropped: int = 0
    duplicate_dates_dropped: int = 0
    missing_days_added: int = 0
    non_numeric_values_blanked: int = 0
    negative_values_clipped: int = 0
    cells_filled: int = 0


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a cleaned copy of `df` and a report of what was changed.

    The result has one row per calendar day, sorted by date, with a
    datetime `Date` column and float numeric columns. Raises SchemaError
    if a required column is missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {', '.join(missing)}")

    report = CleaningReport(rows_in=len(df))
    out = df[REQUIRED_COLUMNS].copy()

    # 1. Parse dates; rows with no usable date can't be placed on the calendar.
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    bad_dates = out[DATE_COLUMN].isna()
    report.unparseable_date_rows_dropped = int(bad_dates.sum())
    out = out[~bad_dates]

    # 2. Numeric columns: text like "N/A" becomes a blank to be filled later.
    for col in NUMERIC_COLUMNS:
        before_blank = out[col].isna()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        report.non_numeric_values_blanked += int((out[col].isna() & ~before_blank).sum())

    # 3. Duplicates: identical rows first, then repeated dates (keep first).
    n = len(out)
    out = out.drop_duplicates()
    report.duplicate_rows_dropped = n - len(out)
    n = len(out)
    out = out.drop_duplicates(subset=DATE_COLUMN, keep="first")
    report.duplicate_dates_dropped = n - len(out)

    if out.empty:
        report.rows_out = 0
        return out.reset_index(drop=True), report

    # 4. Reindex to a complete daily calendar (new days start as blanks).
    out = out.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    full_range = pd.date_range(out.index.min(), out.index.max(), freq="D")
    report.missing_days_added = len(full_range) - len(out)
    out = out.reindex(full_range)
    out.index.name = DATE_COLUMN

    # 5. Impossible negatives -> 0. Real zeros and large values are left alone.
    report.negative_values_clipped = int((out[NUMERIC_COLUMNS] < 0).sum().sum())
    out[NUMERIC_COLUMNS] = out[NUMERIC_COLUMNS].clip(lower=0)

    # 6. Fill remaining blanks (nulls, non-numeric, new days) from neighbors.
    report.cells_filled = int(out[NUMERIC_COLUMNS].isna().sum().sum())
    out[NUMERIC_COLUMNS] = out[NUMERIC_COLUMNS].ffill().bfill()

    out = out.reset_index()
    report.rows_out = len(out)
    return out, report
