"""Corrupt a copy of the synthetic KPI data with data-quality problems.

This is separate from generate_synthetic_data.py on purpose: that script
injects *business anomalies* (real events the detector should find), while
this one injects *data-quality defects* (errors the cleaner should fix).

Input:  data/raw/ecommerce_daily_kpis.csv          (left untouched)
Output: data/raw/ecommerce_daily_kpis_messy.csv

Defects injected (each on a different, interior row so counts are exact):
  5 missing days, 3 duplicated rows, 2 repeated dates with different values,
  6 null cells, 3 negative values, 2 text values in numeric columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "ecommerce_daily_kpis.csv"
OUTPUT_PATH = ROOT / "data" / "raw" / "ecommerce_daily_kpis_messy.csv"

NUMERIC_COLUMNS = [
    "Revenue",
    "Orders",
    "Website_Traffic",
    "Conversion_Rate",
    "Ad_Spend",
    "Refunds",
    "Average_Order_Value",
]
NEGATIVE_COLUMNS = ["Revenue", "Ad_Spend", "Refunds"]

N_MISSING_DAYS = 5
N_DUP_ROWS = 3
N_DUP_DATES = 2
N_NULLS = 6
N_NEGATIVES = 3
N_TEXT = 2


def make_messy(clean: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a corrupted copy of `clean` (Date + the 7 numeric columns)."""
    rng = np.random.default_rng(seed)
    df = clean.copy()
    df["Date"] = df["Date"].astype(str)
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].astype(object)

    # Distinct interior rows, so first/last dates stay and defects never overlap.
    total = N_MISSING_DAYS + N_DUP_ROWS + N_DUP_DATES + N_NULLS + N_NEGATIVES + N_TEXT
    picks = rng.choice(np.arange(1, len(df) - 1), size=total, replace=False)
    it = iter(picks)
    take = lambda n: [next(it) for _ in range(n)]  # noqa: E731

    missing_rows = take(N_MISSING_DAYS)
    dup_rows = take(N_DUP_ROWS)
    dup_date_rows = take(N_DUP_DATES)
    null_rows = take(N_NULLS)
    neg_rows = take(N_NEGATIVES)
    text_rows = take(N_TEXT)

    for r in null_rows:
        df.at[r, str(rng.choice(NUMERIC_COLUMNS))] = np.nan
    for r in neg_rows:
        col = str(rng.choice(NEGATIVE_COLUMNS))
        df.at[r, col] = -abs(float(df.at[r, col]))
    for r, text in zip(text_rows, ["N/A", "abc"]):
        df.at[r, str(rng.choice(NUMERIC_COLUMNS))] = text

    extras = [df.loc[dup_rows]]  # exact copies
    changed = df.loc[dup_date_rows].copy()  # same date, different Revenue
    changed["Revenue"] = changed["Revenue"].astype(float) * 1.1
    extras.append(changed)

    df = df.drop(index=missing_rows)
    return pd.concat([df] + extras, ignore_index=True)


def main() -> None:
    clean = pd.read_csv(INPUT_PATH)
    messy = make_messy(clean)
    messy.to_csv(OUTPUT_PATH, index=False)
    print(f"Read {len(clean)} rows from {INPUT_PATH.name}")
    print(f"Wrote {len(messy)} rows to {OUTPUT_PATH.name}")
    print(
        f"Injected: {N_MISSING_DAYS} missing days, {N_DUP_ROWS} duplicate rows, "
        f"{N_DUP_DATES} duplicate dates, {N_NULLS} nulls, "
        f"{N_NEGATIVES} negatives, {N_TEXT} text values"
    )


if __name__ == "__main__":
    main()
