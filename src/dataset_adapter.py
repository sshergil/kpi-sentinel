"""Turn an arbitrary uploaded table into KPI Sentinel's internal frame.

Internal representation: a `Date` column plus one numeric column per metric
(wide form), which is what compute_baselines()/clean_data() already take.
The detection algorithms are unchanged; this layer only renames, parses and,
when several rows share a date, aggregates to one row per day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data_loader import DATE_COLUMN
from src.dataset_profiler import SUM_ROLES, classify_role, describe_time_axis, parse_dates, to_numeric

AGGREGATIONS = ("auto", "sum", "mean", "median")


class UnsupportedDatasetError(ValueError):
    """The dataset cannot be analyzed; the message is written for the end user."""


@dataclass
class NormalizedData:
    frame: pd.DataFrame  # Date + metric columns
    metrics: list[str]  # metric column names as they appear in `frame`
    notes: list[str] = field(default_factory=list)
    aggregation: dict[str, str] = field(default_factory=dict)  # metric -> function, if aggregated
    invalid_dates_dropped: int = 0
    non_numeric_blanked: int = 0


def normalize_dataset(
    raw: pd.DataFrame, date_column: str, metrics: list[str], aggregation: str = "auto"
) -> NormalizedData:
    """Parse the time column and metrics; aggregate to one row per day if needed."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
    if date_column not in raw.columns:
        raise UnsupportedDatasetError(f"Time column '{date_column}' was not found in the dataset.")
    metrics = [m for m in metrics if m != date_column]
    unknown = [m for m in metrics if m not in raw.columns]
    if unknown:
        raise UnsupportedDatasetError(f"Metric column(s) not found: {', '.join(unknown)}")
    if not metrics:
        raise UnsupportedDatasetError("Select at least one numeric metric to analyze.")

    axis = describe_time_axis(raw, date_column)
    if axis.problem:
        raise UnsupportedDatasetError(axis.problem)

    notes: list[str] = []
    frame = pd.DataFrame({DATE_COLUMN: parse_dates(raw[date_column])})
    internal: dict[str, str] = {}  # original name -> name used in `frame`
    blanked = 0
    for m in metrics:
        name = f"{m} (metric)" if m == DATE_COLUMN else m
        internal[m] = name
        numeric = to_numeric(raw[m])
        blanked += int((numeric.isna() & raw[m].notna()).sum())
        frame[name] = numeric.to_numpy()
    names = [internal[m] for m in metrics]

    invalid = int(frame[DATE_COLUMN].isna().sum())
    if invalid:
        frame = frame.dropna(subset=[DATE_COLUMN])
        notes.append(f"{invalid} row(s) with a missing or invalid {date_column} were dropped")
    if blanked:
        notes.append(f"{blanked} value(s) were not numeric and were treated as missing")

    used: dict[str, str] = {}
    if axis.multiple_rows_per_date:
        before = len(frame)
        funcs = {n: _choose_function(m, aggregation) for m, n in internal.items()}
        frame = _aggregate_by_date(frame, funcs)
        used = funcs
        summary = ", ".join(f"{f} for {', '.join(c for c in funcs if funcs[c] == f)}" for f in sorted(set(funcs.values())))
        notes.append(f"{before} rows were aggregated into {len(frame)} daily rows ({summary})")

    frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    return NormalizedData(frame, names, notes, used, invalid, blanked)


def _choose_function(metric: str, aggregation: str) -> str:
    if aggregation != "auto":
        return aggregation
    return "sum" if classify_role(metric) in SUM_ROLES else "mean"


def _aggregate_by_date(frame: pd.DataFrame, funcs: dict[str, str]) -> pd.DataFrame:
    grouped = frame.groupby(DATE_COLUMN)
    parts = []
    for func in sorted(set(funcs.values())):
        cols = [c for c, f in funcs.items() if f == func]
        parts.append(grouped[cols].sum(min_count=1) if func == "sum" else getattr(grouped[cols], func)())
    out = pd.concat(parts, axis=1)[list(funcs)]
    return out.reset_index()
