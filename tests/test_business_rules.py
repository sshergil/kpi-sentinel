import pandas as pd

from src.business_rules import classify_incidents


def classify(directions):
    incidents = pd.DataFrame([{"Incident_ID": 1, "Metric_Directions": directions}])
    return classify_incidents(incidents).iloc[0]


def test_demand_spike_is_positive():
    r = classify({"Website_Traffic": "spike", "Conversion_Rate": "spike",
                  "Orders": "spike", "Revenue": "spike"})
    assert r["Category"] == "demand_spike" and r["Impact"] == "positive"


def test_traffic_spike_without_order_lift_is_traffic_quality():
    r = classify({"Website_Traffic": "spike"})
    assert r["Category"] == "traffic_quality" and r["Impact"] == "negative"


def test_conversion_drop_with_stable_traffic():
    r = classify({"Conversion_Rate": "drop", "Orders": "drop", "Revenue": "drop"})
    assert r["Category"] == "conversion_drop"


def test_traffic_drop_with_falling_sales():
    r = classify({"Website_Traffic": "drop", "Orders": "drop", "Revenue": "drop"})
    assert r["Category"] == "traffic_drop"


def test_traffic_and_conversion_both_down_prefers_traffic_drop():
    r = classify({"Website_Traffic": "drop", "Conversion_Rate": "drop", "Orders": "drop"})
    assert r["Category"] == "traffic_drop"


def test_ad_spend_jump_with_traffic_spike_is_marketing_efficiency_not_traffic_quality():
    r = classify({"Ad_Spend": "spike", "Website_Traffic": "spike"})
    assert r["Category"] == "marketing_efficiency"


def test_ad_spend_jump_with_flat_revenue_is_marketing_efficiency():
    assert classify({"Ad_Spend": "spike"})["Category"] == "marketing_efficiency"


def test_ad_spend_jump_with_ratio_drop_is_marketing_efficiency_even_if_revenue_rose():
    r = classify({"Ad_Spend": "spike", "Revenue": "spike", "Revenue_per_Ad_Dollar": "drop"})
    assert r["Category"] == "marketing_efficiency"


def test_ad_spend_jump_with_proportional_revenue_is_not_efficiency_decline():
    r = classify({"Ad_Spend": "spike", "Revenue": "spike"})
    assert r["Category"] == "unclassified"


def test_refund_spike_is_refund_issue():
    assert classify({"Refunds": "spike"})["Category"] == "refund_issue"


def test_refund_rate_alone_counts_but_not_when_revenue_collapsed():
    assert classify({"Refund_Rate": "spike"})["Category"] == "refund_issue"
    r = classify({"Refund_Rate": "spike", "Revenue": "drop"})
    assert r["Category"] == "unclassified"  # ratio rose only because revenue fell


def test_conversion_drop_wins_over_mechanical_refund_rate_flag():
    r = classify({"Conversion_Rate": "drop", "Orders": "drop", "Revenue": "drop",
                  "Refund_Rate": "spike"})
    assert r["Category"] == "conversion_drop"


def test_aov_spike_without_more_orders():
    r = classify({"Average_Order_Value": "spike", "Revenue": "spike"})
    assert r["Category"] == "aov_spike" and r["Impact"] == "positive"


def test_unknown_pattern_is_unclassified():
    r = classify({"Orders": "drop"})
    assert r["Category"] == "unclassified" and r["Impact"] == "neutral"


def test_evidence_lists_flagged_metrics_and_directions():
    r = classify({"Revenue": "drop", "Orders": "drop"})
    assert r["Evidence"] == "Orders drop, Revenue drop"


def test_descriptions_are_hedged_not_causal():
    r = classify({"Conversion_Rate": "drop", "Orders": "drop"})
    assert "consistent with" in r["Pattern"]


def test_input_is_not_modified_and_empty_input_works():
    incidents = pd.DataFrame([{"Incident_ID": 1, "Metric_Directions": {"Refunds": "spike"}}])
    before = incidents.copy()
    classify_incidents(incidents)
    pd.testing.assert_frame_equal(incidents, before)
    empty = pd.DataFrame({"Incident_ID": [], "Metric_Directions": []})
    assert classify_incidents(empty).empty
