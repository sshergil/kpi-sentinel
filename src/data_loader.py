"""Load and validate raw KPI data.

This module only *reads and inspects* data. It never modifies values:
fixing problems is the job of src/data_cleaning.py, and judging whether
unusual values are anomalies is the job of the detection stage.

Only missing required columns are fatal (SchemaError). Every other
problem is recorded as a warning in the returned ValidationReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATE_COLUMN = "Date"
NUMERIC_COLUMNS = [
    "Revenue",
    "Orders",
    "Website_Traffic",
    "Conversion_Rate",
    "Ad_Spend",
    "Refunds",
    "Average_Order_Value",
]
REQUIRED_COLUMNS = [DATE_COLUMN] + NUMERIC_COLUMNS

# Enough history for a rolling baseline (e.g. 28 days) plus data to test on.
MIN_ROWS = 60


class SchemaError(Exception):
    """Raised when the data cannot be used at all (missing required columns)."""


@dataclass
class ValidationReport:
    """Non-fatal findings from validating a dataset."""

    n_rows: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.warnings


def validate_schema(df: pd.DataFrame) -> ValidationReport:
    """Check a raw DataFrame and return a report. Does not modify `df`.

    Raises SchemaError if any required column is missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {', '.join(missing)}")

    report = ValidationReport(n_rows=len(df))
    warn = report.warnings.append

    # Dates: work on a parsed copy so the original is untouched.
    dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    n_bad_dates = int(dates.isna().sum())
    if n_bad_dates:
        warn(f"{n_bad_dates} row(s) have a missing or unparseable {DATE_COLUMN}")

    valid_dates = dates.dropna()
    n_dup_dates = int(valid_dates.duplicated().sum())
    if n_dup_dates:
        warn(f"{n_dup_dates} duplicate date(s)")

    if not valid_dates.empty:
        expected_days = (valid_dates.max() - valid_dates.min()).days + 1
        n_missing_days = expected_days - valid_dates.nunique()
        if n_missing_days:
            warn(f"{n_missing_days} missing calendar day(s) in the date range")

    n_dup_rows = int(df.duplicated().sum())
    if n_dup_rows:
        warn(f"{n_dup_rows} fully duplicated row(s)")

    # Numeric columns: distinguish nulls from non-numeric text and negatives.
    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_null = int(df[col].isna().sum())
        n_non_numeric = int((coerced.isna() & df[col].notna()).sum())
        n_negative = int((coerced < 0).sum())
        if n_null:
            warn(f"{col}: {n_null} missing value(s)")
        if n_non_numeric:
            warn(f"{col}: {n_non_numeric} non-numeric value(s)")
        if n_negative:
            warn(f"{col}: {n_negative} negative value(s)")

    if len(valid_dates) < MIN_ROWS:
        warn(
            f"Only {len(valid_dates)} dated row(s); at least {MIN_ROWS} "
            "recommended for a reliable rolling baseline"
        )

    return report


def load_data(path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Read a CSV or Excel file and validate it.

    Returns the data exactly as read (no cleaning, no type conversion)
    together with a ValidationReport.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .csv, .xlsx, or .xls")

    return df, validate_schema(df)
