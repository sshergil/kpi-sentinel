import pandas as pd

from src.severity import score_incidents


def score(peak_z=5.0, days=1, directions=None, impact="negative"):
    directions = directions if directions is not None else {"Orders": "drop"}
    incidents = pd.DataFrame([{
        "Incident_ID": 1, "Peak_Abs_Z": peak_z, "Duration_Days": days,
        "Metrics": sorted(directions), "Metric_Directions": directions, "Impact": impact,
    }])
    return score_incidents(incidents).iloc[0]


def test_larger_z_scores_higher():
    assert score(peak_z=9.0)["Severity_Score"] > score(peak_z=5.0)["Severity_Score"]


def test_longer_incident_scores_higher():
    assert score(days=4)["Severity_Score"] > score(days=1)["Severity_Score"]


def test_more_metrics_score_higher():
    wide = {"Orders": "drop", "Conversion_Rate": "drop", "Website_Traffic": "drop"}
    assert score(directions=wide)["Severity_Score"] > score()["Severity_Score"]


def test_single_metric_incident_is_damped():
    one = score(directions={"Revenue": "drop"})["Severity_Score"]
    two = score(directions={"Revenue": "drop", "Orders": "drop"})["Severity_Score"]
    assert two > one
    assert "weak single-metric flag" in score(directions={"Revenue": "drop"})["Severity_Breakdown"]


def test_extreme_single_metric_flag_is_not_damped():
    weak = score(peak_z=5.0, directions={"Refunds": "spike"})
    strong = score(peak_z=9.0, directions={"Refunds": "spike"})
    assert "weak single-metric" in weak["Severity_Breakdown"]
    assert "weak single-metric" not in strong["Severity_Breakdown"]
    assert strong["Severity_Score"] > 2 * weak["Severity_Score"]


def test_derived_metrics_do_not_add_breadth():
    base = score(directions={"Revenue": "drop"})["Severity_Score"]
    with_ratio = score(directions={"Revenue": "drop", "Refund_Rate": "spike"})["Severity_Score"]
    assert with_ratio == base


def test_revenue_drop_outranks_revenue_spike_outranks_none():
    drop = score(directions={"Revenue": "drop"})["Severity_Score"]
    spike = score(directions={"Revenue": "spike"})["Severity_Score"]
    none = score(directions={"Orders": "drop"})["Severity_Score"]
    assert drop > spike > none


def test_positive_impact_is_scaled_down():
    d = {"Revenue": "spike", "Orders": "spike"}
    assert score(directions=d, impact="positive")["Severity_Score"] < score(directions=d)["Severity_Score"]


def test_components_are_capped_and_score_stays_in_range():
    huge = score(peak_z=1000.0, days=30,
                 directions={m: "drop" for m in ["Revenue", "Orders", "Conversion_Rate", "Website_Traffic", "Refunds"]})
    assert huge["Severity_Score"] == 100.0
    assert huge["Severity_Label"] == "Critical"
    assert score(peak_z=4.0)["Severity_Score"] >= 0


def test_label_boundaries():
    assert score(peak_z=4.0)["Severity_Label"] == "Low"
    strong = score(peak_z=10.0, days=3, directions={"Revenue": "drop", "Orders": "drop", "Conversion_Rate": "drop"})
    assert strong["Severity_Label"] in ("High", "Critical")


def test_breakdown_explains_the_score():
    b = score()["Severity_Breakdown"]
    assert "magnitude" in b and "breadth" in b and "duration" in b and "revenue" in b


def test_input_not_modified_and_empty_input_works():
    incidents = pd.DataFrame([{
        "Incident_ID": 1, "Peak_Abs_Z": 5.0, "Duration_Days": 1,
        "Metrics": ["Orders"], "Metric_Directions": {"Orders": "drop"},
    }])
    before = incidents.copy()
    score_incidents(incidents)
    pd.testing.assert_frame_equal(incidents, before)
    assert score_incidents(incidents.iloc[0:0]).empty
