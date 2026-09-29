import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    MIN_ROWS,
    NUMERIC_COLUMNS,
    SchemaError,
    load_data,
    validate_schema,
)


@pytest.fixture
def clean_df():
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=100)})
    for col in NUMERIC_COLUMNS:
        df[col] = np.arange(100, dtype=float) + 10
    return df


def has_warning(report, text):
    return any(text in w for w in report.warnings)


def test_clean_data_has_no_warnings(clean_df):
    report = validate_schema(clean_df)
    assert report.is_clean
    assert report.n_rows == 100


def test_missing_column_is_fatal(clean_df):
    with pytest.raises(SchemaError, match="Refunds"):
        validate_schema(clean_df.drop(columns=["Refunds"]))


def test_missing_date_column_is_fatal(clean_df):
    with pytest.raises(SchemaError, match="Date"):
        validate_schema(clean_df.drop(columns=["Date"]))


def test_non_numeric_values_warn(clean_df):
    clean_df["Orders"] = clean_df["Orders"].astype(object)
    clean_df.loc[3, "Orders"] = "abc"
    report = validate_schema(clean_df)
    assert has_warning(report, "Orders: 1 non-numeric")


def test_unparseable_dates_warn(clean_df):
    clean_df["Date"] = clean_df["Date"].astype(str)
    clean_df.loc[2, "Date"] = "not-a-date"
    report = validate_schema(clean_df)
    assert has_warning(report, "1 row(s) have a missing or unparseable Date")


def test_insufficient_history_warns(clean_df):
    report = validate_schema(clean_df.head(10))
    assert has_warning(report, f"at least {MIN_ROWS}")


def test_duplicate_rows_warn(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    report = validate_schema(df)
    assert has_warning(report, "1 fully duplicated row")
    assert has_warning(report, "1 duplicate date")


def test_duplicate_dates_with_different_values_warn(clean_df):
    dup = clean_df.iloc[[5]].copy()
    dup["Revenue"] = 999.0
    df = pd.concat([clean_df, dup], ignore_index=True)
    report = validate_schema(df)
    assert has_warning(report, "1 duplicate date")
    assert not has_warning(report, "fully duplicated")


def test_missing_calendar_days_warn(clean_df):
    report = validate_schema(clean_df.drop(index=[10, 11]))
    assert has_warning(report, "2 missing calendar day")


def test_missing_values_warn(clean_df):
    clean_df.loc[4, "Ad_Spend"] = np.nan
    report = validate_schema(clean_df)
    assert has_warning(report, "Ad_Spend: 1 missing")


def test_negative_values_warn(clean_df):
    clean_df.loc[7, "Revenue"] = -5.0
    report = validate_schema(clean_df)
    assert has_warning(report, "Revenue: 1 negative")


def test_validation_does_not_modify_input(clean_df):
    clean_df["Orders"] = clean_df["Orders"].astype(object)
    clean_df.loc[3, "Orders"] = "abc"
    before = clean_df.copy()
    validate_schema(clean_df)
    pd.testing.assert_frame_equal(clean_df, before)


def test_load_data_reads_csv(clean_df, tmp_path):
    path = tmp_path / "kpis.csv"
    clean_df.to_csv(path, index=False)
    df, report = load_data(path)
    assert len(df) == 100
    assert report.is_clean


def test_load_data_rejects_unknown_extension(tmp_path):
    path = tmp_path / "kpis.txt"
    path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        load_data(path)
