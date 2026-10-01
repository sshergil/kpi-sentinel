import numpy as np
import pandas as pd

from src.data_loader import NUMERIC_COLUMNS
from src.pipeline import run_detection_pipeline

START = pd.Timestamp("2025-01-01")


def synthetic(n=160, seed=0):
    """Seasonal data (weekends higher), 3% independent noise per metric."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=n)
    weekly = np.where(dates.dayofweek >= 5, 1.25, 1.0)
    levels = {"Revenue": 20000, "Orders": 400, "Website_Traffic": 12000,
              "Conversion_Rate": 0.033, "Ad_Spend": 3000, "Refunds": 600,
              "Average_Order_Value": 50}
    df = pd.DataFrame({"Date": dates})
    for col in NUMERIC_COLUMNS:
        df[col] = levels[col] * weekly * rng.normal(1, 0.03, n)
    return df


def inject(df, day, **factors):
    for col, factor in factors.items():
        df.loc[day, col] *= factor
    return df


def incident_on(incidents, day):
    date = START + pd.Timedelta(days=day)
    hit = incidents[(incidents["Start_Date"] <= date) & (incidents["End_Date"] >= date)]
    assert len(hit) >= 1, f"no incident on day {day}"
    return hit.iloc[0]


def test_conversion_drop_is_found_classified_and_scored_high():
    df = inject(synthetic(), 100, Conversion_Rate=0.5, Orders=0.5, Revenue=0.5)
    _, incidents = run_detection_pipeline(df)
    inc = incident_on(incidents, 100)
    assert inc["Category"] == "conversion_drop"
    assert inc["Severity_Label"] in ("High", "Critical")


def test_refund_spike_is_classified_as_refund_issue():
    df = inject(synthetic(), 120, Refunds=4.0)
    _, incidents = run_detection_pipeline(df)
    assert incident_on(incidents, 120)["Category"] == "refund_issue"


def test_demand_spike_is_classified_positive():
    df = inject(synthetic(), 135, Website_Traffic=1.8, Conversion_Rate=1.3, Orders=2.3, Revenue=2.3)
    _, incidents = run_detection_pipeline(df)
    inc = incident_on(incidents, 135)
    assert inc["Category"] == "demand_spike" and inc["Impact"] == "positive"


def test_ratio_metrics_catch_efficiency_decline():
    # Ad spend doubles while revenue stays flat: only the ratio metric (and
    # Ad_Spend itself) can show this; revenue alone looks normal.
    df = inject(synthetic(), 110, Ad_Spend=2.0)
    detections, incidents = run_detection_pipeline(df)
    day = detections[detections["Date"] == START + pd.Timedelta(days=110)]
    assert day.loc[day["Metric"] == "Revenue_per_Ad_Dollar", "Is_Anomaly"].iloc[0]
    assert incident_on(incidents, 110)["Category"] == "marketing_efficiency"


def test_injected_event_outranks_all_noise_incidents():
    df = inject(synthetic(), 100, Conversion_Rate=0.5, Orders=0.5, Revenue=0.5)
    _, incidents = run_detection_pipeline(df)
    top = incidents.sort_values("Severity_Score", ascending=False).iloc[0]
    assert top["Start_Date"] <= START + pd.Timedelta(days=100) <= top["End_Date"]


def test_quiet_data_never_produces_critical_incidents():
    for seed in range(5):
        _, incidents = run_detection_pipeline(synthetic(seed=seed))
        assert not (incidents["Severity_Label"] == "Critical").any()


def test_pipeline_does_not_modify_input():
    df = synthetic()
    before = df.copy()
    run_detection_pipeline(df)
    pd.testing.assert_frame_equal(df, before)
