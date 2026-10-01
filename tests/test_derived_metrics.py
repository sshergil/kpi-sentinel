import numpy as np
import pandas as pd
import pytest

from src.derived_metrics import ALL_METRICS, DERIVED_COLUMNS, add_derived_metrics
from src.data_loader import NUMERIC_COLUMNS


def frame():
    return pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=3),
        "Revenue": [1000.0, 2000.0, 0.0],
        "Refunds": [50.0, 100.0, 10.0],
        "Ad_Spend": [200.0, 0.0, 100.0],
    })


def test_ratios_are_computed():
    out = add_derived_metrics(frame())
    assert out.loc[0, "Refund_Rate"] == pytest.approx(0.05)
    assert out.loc[0, "Revenue_per_Ad_Dollar"] == pytest.approx(5.0)
    assert out.loc[1, "Refund_Rate"] == pytest.approx(0.05)


def test_zero_denominator_gives_nan_not_inf():
    out = add_derived_metrics(frame())
    assert np.isnan(out.loc[1, "Revenue_per_Ad_Dollar"])  # Ad_Spend = 0
    assert np.isnan(out.loc[2, "Refund_Rate"])            # Revenue = 0
    assert not np.isinf(out[DERIVED_COLUMNS].to_numpy(dtype=float)).any()


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Ad_Spend"):
        add_derived_metrics(frame().drop(columns=["Ad_Spend"]))


def test_input_is_not_modified():
    df = frame()
    before = df.copy()
    add_derived_metrics(df)
    pd.testing.assert_frame_equal(df, before)


def test_all_metrics_lists_raw_then_derived():
    assert ALL_METRICS == list(NUMERIC_COLUMNS) + DERIVED_COLUMNS
