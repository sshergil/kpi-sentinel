import pandas as pd
import pytest

from src.incidents import INCIDENT_COLUMNS, build_incidents


def cells(rows):
    """rows: (date, metric, z, direction or None). Direction set => flagged."""
    return pd.DataFrame([
        {"Date": pd.Timestamp(d), "Metric": m, "Z_Score": z,
         "Direction": direction, "Is_Anomaly": direction is not None}
        for d, m, z, direction in rows
    ])


def test_same_day_flags_across_metrics_form_one_incident():
    out = build_incidents(cells([
        ("2025-03-01", "Revenue", -6.0, "drop"),
        ("2025-03-01", "Orders", -5.0, "drop"),
        ("2025-03-01", "Refunds", 0.2, None),
    ]))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Metrics"] == ["Orders", "Revenue"]
    assert row["Metric_Directions"] == {"Orders": "drop", "Revenue": "drop"}
    assert row["N_Flagged_Cells"] == 2
    assert row["Peak_Abs_Z"] == 6.0
    assert row["Duration_Days"] == 1


def test_consecutive_days_merge_and_gaps_split():
    out = build_incidents(cells([
        ("2025-03-01", "Revenue", -5.0, "drop"),
        ("2025-03-02", "Orders", -5.0, "drop"),
        ("2025-03-03", "Revenue", -5.0, "drop"),
        ("2025-03-06", "Refunds", 5.0, "spike"),  # 3-day gap -> new incident
    ]))
    assert len(out) == 2
    first, second = out.iloc[0], out.iloc[1]
    assert first["Start_Date"] == pd.Timestamp("2025-03-01")
    assert first["End_Date"] == pd.Timestamp("2025-03-03")
    assert first["Duration_Days"] == 3
    assert second["Duration_Days"] == 1
    assert list(out["Incident_ID"]) == [1, 2]


def test_max_gap_days_is_configurable():
    rows = [("2025-03-01", "Revenue", -5.0, "drop"), ("2025-03-03", "Orders", -5.0, "drop")]
    assert len(build_incidents(cells(rows), max_gap_days=1)) == 2
    assert len(build_incidents(cells(rows), max_gap_days=2)) == 1  # any metric, 2 days apart


def test_metric_moving_both_ways_is_mixed():
    out = build_incidents(cells([
        ("2025-03-01", "Revenue", 5.0, "spike"),
        ("2025-03-02", "Revenue", -5.0, "drop"),
    ]))
    assert out.iloc[0]["Metric_Directions"] == {"Revenue": "mixed"}


def test_no_flags_gives_empty_table_with_columns():
    out = build_incidents(cells([("2025-03-01", "Revenue", 0.1, None)]))
    assert out.empty
    assert list(out.columns) == INCIDENT_COLUMNS


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="detect_anomalies"):
        build_incidents(pd.DataFrame({"Date": []}))


def test_one_day_gap_is_bridged_when_runs_share_a_metric():
    out = build_incidents(cells([
        ("2025-07-25", "Refunds", 6.0, "spike"),
        ("2025-07-27", "Refunds", 5.0, "spike"),
        ("2025-07-28", "Refunds", 5.0, "spike"),
    ]))
    assert len(out) == 1
    assert out.iloc[0]["Start_Date"] == pd.Timestamp("2025-07-25")
    assert out.iloc[0]["End_Date"] == pd.Timestamp("2025-07-28")
    assert out.iloc[0]["Duration_Days"] == 4


def test_gap_is_not_bridged_when_runs_share_no_metric():
    out = build_incidents(cells([
        ("2025-02-09", "Refunds", 6.0, "spike"),
        ("2025-02-11", "Website_Traffic", -5.0, "drop"),
    ]))
    assert len(out) == 2


def test_gap_longer_than_bridge_is_not_merged_and_bridging_can_be_disabled():
    far = cells([("2025-03-01", "Revenue", -5.0, "drop"), ("2025-03-04", "Revenue", -5.0, "drop")])
    assert len(build_incidents(far)) == 2
    near = cells([("2025-03-01", "Revenue", -5.0, "drop"), ("2025-03-03", "Revenue", -5.0, "drop")])
    assert len(build_incidents(near)) == 1
    assert len(build_incidents(near, bridge_gap_days=1)) == 2
