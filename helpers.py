"""Shared builders for upload-related tests.

`upload_frame` makes a daily dataset with arbitrary column names (the kind of
thing a user might upload); `synthetic_kpis` makes data in the built-in
e-commerce schema. Both are seasonal (weekends higher) with independent
multiplicative noise and are fully deterministic for a given seed.

Even pure noise trips the detector on roughly 1-2% of cells (8-sample baselines), and
a chance flag next to an injected event merges into its incident. The default seed is
fixed to one where that does not happen near the days the tests inject events on; most
seeds work (9 of the first 12 do), so the tests do not depend on a lucky draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import NUMERIC_COLUMNS

START = pd.Timestamp("2025-01-01")

ECOM_LIKE = {"Sales": 12000.0, "Customers": 450.0, "Returns": 35.0, "Marketing_Spend": 1800.0}
WEATHER = {"Temperature": 20.0, "Humidity": 55.0, "Pressure": 1013.0}


def upload_frame(
    levels: dict[str, float],
    date_name: str = "Date",
    n: int = 180,
    seasonal: bool = True,
    noise: float = 0.03,
    seed: int = 1,
) -> pd.DataFrame:
    """Daily data: one column per entry in `levels` (its typical value), starting 2025-01-01."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=n)
    weekly = np.where(dates.dayofweek >= 5, 1.25, 1.0) if seasonal else np.ones(n)
    df = pd.DataFrame({date_name: dates})
    for name, level in levels.items():
        df[name] = level * weekly * (rng.normal(1, noise, n) if noise else np.ones(n))
    return df


def synthetic_kpis(conversion_drop_day: int | None = None, n: int = 160, seed: int = 1) -> pd.DataFrame:
    """Built-in e-commerce schema; optionally a one-day conversion collapse (conversion, orders, revenue halve)."""
    levels = {"Revenue": 20000, "Orders": 400, "Website_Traffic": 12000, "Conversion_Rate": 0.033,
              "Ad_Spend": 3000, "Refunds": 600, "Average_Order_Value": 50}
    df = upload_frame({c: levels[c] for c in NUMERIC_COLUMNS}, n=n, seed=seed)
    if conversion_drop_day is not None:
        for col in ("Conversion_Rate", "Orders", "Revenue"):
            df.loc[conversion_drop_day, col] *= 0.5
    return df
