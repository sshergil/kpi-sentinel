import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import clean_data
from src.data_loader import NUMERIC_COLUMNS, SchemaError


@pytest.fixture
def clean_df():
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=100)})
    for col in NUMERIC_COLUMNS:
        df[col] = np.arange(100, dtype=float) + 10
    return df


def test_already_clean_data_is_unchanged(clean_df):
    out, report = clean_data(clean_df)
    pd.testing.assert_frame_equal(out, clean_df)
    assert report.rows_in == report.rows_out == 100
    assert report.cells_filled == 0


def test_missing_column_raises(clean_df):
    with pytest.raises(SchemaError):
        clean_data(clean_df.drop(columns=["Orders"]))


def test_duplicate_rows_removed(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[[3]]], ignore_index=True)
    out, report = clean_data(df)
    assert len(out) == 100
    assert report.duplicate_rows_dropped == 1


def test_duplicate_dates_keep_first(clean_df):
    dup = clean_df.iloc[[5]].copy()
    dup["Revenue"] = 999.0
    df = pd.concat([clean_df, dup], ignore_index=True)
    out, report = clean_data(df)
    assert len(out) == 100
    assert report.duplicate_dates_dropped == 1
    assert out.loc[5, "Revenue"] == clean_df.loc[5, "Revenue"]


def test_missing_calendar_days_are_filled(clean_df):
    out, report = clean_data(clean_df.drop(index=[10, 11]))
    assert len(out) == 100
    assert report.missing_days_added == 2
    # Filled from the previous day (forward fill).
    assert out.loc[10, "Revenue"] == clean_df.loc[9, "Revenue"]
    assert out[NUMERIC_COLUMNS].isna().sum().sum() == 0


def test_missing_values_are_filled(clean_df):
    clean_df.loc[4, "Ad_Spend"] = np.nan
    out, report = clean_data(clean_df)
    assert out.loc[4, "Ad_Spend"] == clean_df.loc[3, "Ad_Spend"]
    assert report.cells_filled == 1


def test_leading_missing_value_is_back_filled(clean_df):
    clean_df.loc[0, "Refunds"] = np.nan
    out, _ = clean_data(clean_df)
    assert out.loc[0, "Refunds"] == clean_df.loc[1, "Refunds"]


def test_text_in_numeric_column_is_fixed(clean_df):
    clean_df["Orders"] = clean_df["Orders"].astype(object)
    clean_df.loc[3, "Orders"] = "N/A"
    out, report = clean_data(clean_df)
    assert report.non_numeric_values_blanked == 1
    assert out["Orders"].dtype == float
    assert out.loc[3, "Orders"] == clean_df.loc[2, "Orders"]


def test_negative_values_are_clipped_to_zero(clean_df):
    clean_df.loc[7, "Revenue"] = -50.0
    out, report = clean_data(clean_df)
    assert out.loc[7, "Revenue"] == 0
    assert report.negative_values_clipped == 1


def test_unparseable_date_rows_are_dropped(clean_df):
    clean_df["Date"] = clean_df["Date"].astype(str)
    clean_df.loc[50, "Date"] = "garbage"
    out, report = clean_data(clean_df)
    assert report.unparseable_date_rows_dropped == 1
    # The dropped day comes back as a missing calendar day and is filled.
    assert len(out) == 100
    assert report.missing_days_added == 1


def test_extreme_but_valid_values_are_not_altered(clean_df):
    clean_df.loc[30, "Revenue"] = 1e9  # huge spike: an anomaly, not a data error
    clean_df.loc[31, "Website_Traffic"] = 1.0  # huge drop
    out, _ = clean_data(clean_df)
    assert out.loc[30, "Revenue"] == 1e9
    assert out.loc[31, "Website_Traffic"] == 1.0


def test_genuine_zero_values_are_not_altered(clean_df):
    clean_df.loc[40, "Orders"] = 0.0  # a real zero-order day
    out, report = clean_data(clean_df)
    assert out.loc[40, "Orders"] == 0.0
    assert report.negative_values_clipped == 0
    assert report.cells_filled == 0


def test_input_is_not_modified(clean_df):
    clean_df["Orders"] = clean_df["Orders"].astype(object)
    clean_df.loc[3, "Orders"] = "N/A"
    clean_df.loc[7, "Revenue"] = -50.0
    before = clean_df.copy()
    clean_data(clean_df)
    pd.testing.assert_frame_equal(clean_df, before)


def test_output_is_sorted_daily_calendar(clean_df):
    shuffled = clean_df.sample(frac=1, random_state=0)
    out, _ = clean_data(shuffled)
    assert out["Date"].is_monotonic_increasing
    assert (out["Date"].diff().dropna() == pd.Timedelta(days=1)).all()
