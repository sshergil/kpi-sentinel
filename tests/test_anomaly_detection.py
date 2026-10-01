import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import Z_THRESHOLD, detect_anomalies
from src.baselines import compute_baselines
from src.data_loader import NUMERIC_COLUMNS


def baseline_row(value, mean=100.0, z=0.0, q1=98.0, q3=102.0):
    """One hand-built row in the shape compute_baselines() returns."""
    return {
        "Value": value, "Baseline_Mean": mean, "Baseline_Std": 2.0,
        "Baseline_Q1": q1, "Baseline_Q3": q3, "Z_Score": z,
    }


def detect_one(**kwargs):
    return detect_anomalies(pd.DataFrame([baseline_row(**kwargs)])).iloc[0]


def test_spike_flagged_when_both_checks_agree():
    r = detect_one(value=120.0, z=10.0)
    assert r["Z_Flag"] and r["IQR_Flag"] and r["Is_Anomaly"]
    assert r["Direction"] == "spike"


def test_drop_flagged_with_drop_direction():
    r = detect_one(value=80.0, z=-10.0)
    assert r["Is_Anomaly"]
    assert r["Direction"] == "drop"


def test_z_only_is_not_an_anomaly():
    r = detect_one(value=101.0, z=5.0)  # inside IQR fences [92, 108]
    assert r["Z_Flag"] and not r["IQR_Flag"]
    assert not r["Is_Anomaly"]


def test_iqr_only_is_not_an_anomaly():
    r = detect_one(value=120.0, z=1.0)
    assert r["IQR_Flag"] and not r["Z_Flag"]
    assert not r["Is_Anomaly"]


def test_z_threshold_is_strict():
    assert Z_THRESHOLD == 4.0
    r = detect_one(value=120.0, z=Z_THRESHOLD)
    assert not r["Z_Flag"]


def test_rows_without_baseline_are_never_flagged():
    row = baseline_row(1e9, mean=np.nan, z=np.nan, q1=np.nan, q3=np.nan)
    r = detect_anomalies(pd.DataFrame([row])).iloc[0]
    assert not r["Z_Flag"] and not r["IQR_Flag"] and not r["Is_Anomaly"]
    assert pd.isna(r["Direction"])


def test_non_anomaly_has_no_direction():
    assert pd.isna(detect_one(value=100.0, z=0.0)["Direction"])


def test_thresholds_are_configurable():
    df = pd.DataFrame([baseline_row(value=120.0, z=3.5)])
    assert detect_anomalies(df, z_threshold=3.0).iloc[0]["Is_Anomaly"]
    assert not detect_anomalies(df, z_threshold=4.0).iloc[0]["Is_Anomaly"]


def test_invalid_parameters_raise():
    df = pd.DataFrame([baseline_row(value=100.0)])
    with pytest.raises(ValueError, match="z_threshold"):
        detect_anomalies(df, z_threshold=0)
    with pytest.raises(ValueError, match="iqr_multiplier"):
        detect_anomalies(df, iqr_multiplier=-1)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="compute_baselines"):
        detect_anomalies(pd.DataFrame({"Value": [1.0]}))


def test_input_is_not_modified():
    df = pd.DataFrame([baseline_row(value=120.0, z=10.0)])
    before = df.copy()
    detect_anomalies(df)
    pd.testing.assert_frame_equal(df, before)


# --- integration with compute_baselines ---------------------------------

def seasonal_df(n=120, seed=0):
    """Weekdays ~100, weekends ~160, noise sd 3, for every metric."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n)
    base = np.where(dates.dayofweek >= 5, 160.0, 100.0)
    df = pd.DataFrame({"Date": dates})
    for col in NUMERIC_COLUMNS:
        df[col] = base + rng.normal(0, 3, n)
    return df


def flagged_dates(result):
    return set(result.loc[result["Is_Anomaly"], "Date"])


def test_saturday_spike_flagged_but_normal_saturdays_are_not():
    df = seasonal_df()
    spike_day = 59  # a Saturday (2025-01-01 is a Wednesday; 3 + 7*8)
    assert df.loc[spike_day, "Date"].dayofweek == 5
    df.loc[spike_day, "Revenue"] += 100

    result = detect_anomalies(compute_baselines(df, metrics=["Revenue"]))
    spike = result[result["Date"] == df.loc[spike_day, "Date"]].iloc[0]
    assert spike["Is_Anomaly"] and spike["Direction"] == "spike"

    other_saturdays = result[
        (result["Date"].dt.dayofweek == 5)
        & (result["Date"] != df.loc[spike_day, "Date"])
        & result["Z_Score"].notna()
        # Skip the 4 Saturdays after the spike: it legitimately sits in their baseline.
        & ~result["Date"].between(df.loc[spike_day, "Date"], df.loc[spike_day, "Date"] + pd.Timedelta(days=28))
    ]
    assert other_saturdays["Is_Anomaly"].mean() < 0.15


def test_weekend_lift_alone_does_not_trigger_mass_flags():
    result = detect_anomalies(compute_baselines(seasonal_df(), metrics=["Revenue"]))
    scored = result[result["Z_Score"].notna()]
    assert scored["Is_Anomaly"].mean() < 0.15  # ~7% expected from 4-sample noise
    weekends = scored[scored["Date"].dt.dayofweek >= 5]
    assert weekends["Is_Anomaly"].mean() < 0.15


def test_drop_is_flagged_with_drop_direction():
    df = seasonal_df()
    df.loc[70, "Orders"] = 20.0
    result = detect_anomalies(compute_baselines(df, metrics=["Orders"]))
    r = result[result["Date"] == df.loc[70, "Date"]].iloc[0]
    assert r["Is_Anomaly"] and r["Direction"] == "drop"


def test_anomaly_in_one_metric_does_not_flag_others():
    df = seasonal_df()
    df.loc[70, "Revenue"] += 200
    result = detect_anomalies(compute_baselines(df))
    day = result[result["Date"] == df.loc[70, "Date"]]
    assert day.loc[day["Metric"] == "Revenue", "Is_Anomaly"].iloc[0]
    # Other metrics are independent noise; allow chance flags but not a sweep.
    assert day.loc[day["Metric"] != "Revenue", "Is_Anomaly"].sum() <= 2


def test_multi_day_anomaly_is_detected_on_first_day():
    df = seasonal_df()
    df.loc[70:72, "Revenue"] += 150
    result = detect_anomalies(compute_baselines(df, metrics=["Revenue"]))
    assert df.loc[70, "Date"] in flagged_dates(result)
