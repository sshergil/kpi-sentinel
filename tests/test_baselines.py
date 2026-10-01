import numpy as np
import pandas as pd
import pytest

from src.baselines import WINDOW_WEEKS, compute_baselines
from src.data_loader import NUMERIC_COLUMNS


def make_df(values_by_metric=None, n=100):
    """Daily frame; every metric is the day index unless overridden."""
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=n)})
    for col in NUMERIC_COLUMNS:
        df[col] = np.arange(n, dtype=float)
    for col, values in (values_by_metric or {}).items():
        df[col] = values
    return df


def seasonal_values(n=100, seed=0):
    """Weekdays ~100, weekends ~160, small noise: normal weekly seasonality."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n)
    base = np.where(dates.dayofweek >= 5, 160.0, 100.0)
    return base + rng.normal(0, 3, n)


def row(result, metric, day_index):
    date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day_index)
    return result[(result["Metric"] == metric) & (result["Date"] == date)].iloc[0]


def test_default_window_is_eight_weeks():
    assert WINDOW_WEEKS == 8
    r = row(compute_baselines(make_df(), metrics=["Revenue"]), "Revenue", 60)
    prior = [53, 46, 39, 32, 25, 18, 11, 4]
    assert r["Baseline_Mean"] == pytest.approx(np.mean(prior))
    assert r["Baseline_Std"] == pytest.approx(np.std(prior, ddof=1))


def test_baseline_uses_same_weekday_previous_four_weeks():
    result = compute_baselines(make_df(), metrics=["Revenue"], window_weeks=4)
    r = row(result, "Revenue", 40)
    prior = [33, 26, 19, 12]  # 7, 14, 21, 28 days earlier
    assert r["Value"] == 40
    assert r["Baseline_Mean"] == pytest.approx(np.mean(prior))
    assert r["Baseline_Std"] == pytest.approx(np.std(prior, ddof=1))
    assert r["Baseline_Q1"] == pytest.approx(np.quantile(prior, 0.25))
    assert r["Baseline_Q3"] == pytest.approx(np.quantile(prior, 0.75))
    assert r["Z_Score"] == pytest.approx((40 - np.mean(prior)) / np.std(prior, ddof=1))


def test_current_day_does_not_contaminate_its_own_baseline():
    plain = compute_baselines(make_df(), metrics=["Revenue"])
    values = np.arange(100, dtype=float)
    values[60] = 1e6
    spiked = compute_baselines(make_df({"Revenue": values}), metrics=["Revenue"])
    assert row(spiked, "Revenue", 60)["Baseline_Mean"] == row(plain, "Revenue", 60)["Baseline_Mean"]
    assert row(spiked, "Revenue", 60)["Baseline_Std"] == row(plain, "Revenue", 60)["Baseline_Std"]


def test_warmup_period_has_no_baseline():
    result = compute_baselines(make_df(), metrics=["Revenue"])
    warmup = 7 * WINDOW_WEEKS  # 56 days
    assert result.loc[result["Date"] < pd.Timestamp("2025-01-01") + pd.Timedelta(days=warmup), "Baseline_Mean"].isna().all()
    assert not np.isnan(row(result, "Revenue", warmup)["Baseline_Mean"])


def test_one_row_per_day_per_metric():
    result = compute_baselines(make_df())
    assert len(result) == 100 * len(NUMERIC_COLUMNS)
    assert set(result["Metric"]) == set(NUMERIC_COLUMNS)
    assert list(result.columns) == [
        "Date", "Metric", "Value", "Baseline_Mean", "Baseline_Std",
        "Baseline_Q1", "Baseline_Q3", "Z_Score",
    ]


def test_normal_weekend_vs_weekday_difference_is_not_extreme():
    result = compute_baselines(make_df({"Revenue": seasonal_values()}), metrics=["Revenue"])
    z = result["Z_Score"].dropna()
    assert len(z) == 100 - 7 * WINDOW_WEEKS
    # A 60-unit weekend lift on ~3-unit noise would be huge against a plain
    # window. Against same-weekday baselines it should be unremarkable.
    assert (z.abs() > 4).mean() < 0.10
    weekends = result[result["Date"].dt.dayofweek >= 5]["Z_Score"].dropna()
    assert weekends.abs().median() < 1.5


def test_large_saturday_spike_is_flagged():
    values = seasonal_values()
    spike_day = 59  # 2025-01-01 is a Wednesday, so day 3 is a Saturday; 3 + 7*8 = 59
    assert (pd.Timestamp("2025-01-01") + pd.Timedelta(days=spike_day)).dayofweek == 5
    values[spike_day] += 100
    result = compute_baselines(make_df({"Revenue": values}), metrics=["Revenue"])
    assert row(result, "Revenue", spike_day)["Z_Score"] > 10


def test_large_drop_gives_negative_z():
    values = seasonal_values()
    values[60] = 5.0
    result = compute_baselines(make_df({"Revenue": values}), metrics=["Revenue"])
    assert row(result, "Revenue", 60)["Z_Score"] < -10


def test_zero_baseline_std_gives_nan_z():
    result = compute_baselines(make_df({"Revenue": np.full(100, 50.0)}), metrics=["Revenue"])
    r = row(result, "Revenue", 60)
    assert r["Baseline_Std"] == 0
    assert np.isnan(r["Z_Score"])


def test_non_daily_dates_raise():
    df = make_df().drop(index=[10])
    with pytest.raises(ValueError, match="clean_data"):
        compute_baselines(df)


def test_missing_metric_column_raises():
    with pytest.raises(ValueError, match="Nope"):
        compute_baselines(make_df(), metrics=["Nope"])


def test_window_weeks_must_allow_a_std():
    with pytest.raises(ValueError, match="window_weeks"):
        compute_baselines(make_df(), window_weeks=1)


def test_input_is_not_modified():
    df = make_df()
    before = df.copy()
    compute_baselines(df)
    pd.testing.assert_frame_equal(df, before)
