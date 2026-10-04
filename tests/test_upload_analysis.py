import json

import numpy as np
import pandas as pd
import pytest

from src.business_rules import classify_incidents
from src.data_cleaning import clean_data
from src.data_loader import SchemaError, validate_schema
from src.dataset_adapter import UnsupportedDatasetError, normalize_dataset
from src.derived_metrics import ALL_METRICS
from src.incidents import incident_cells
from src.llm_summary import (
    build_incident_facts, summarize_incident, summarize_incidents, template_note, validate_note,
)
from src.service import analyze, analyze_upload
from src.severity import score_incidents
from tests.helpers import ECOM_LIKE, WEATHER, synthetic_kpis, upload_frame


def incident_on(a, day_index, start="2025-01-01"):
    date = pd.Timestamp(start) + pd.Timedelta(days=day_index)
    hit = a.incidents[(a.incidents["Start_Date"] <= date) & (a.incidents["End_Date"] >= date)]
    assert len(hit) >= 1, f"no incident on day {day_index}"
    return hit.iloc[0]


# ---- end to end: arbitrary column names ----------------------------------------------------
def test_arbitrary_ecommerce_columns_run_through_the_existing_engine():
    df = upload_frame(ECOM_LIKE)
    df.loc[100, "Sales"] *= 0.4
    a = analyze_upload(df, "Date", list(ECOM_LIKE))
    assert a.generic and a.metrics == list(ECOM_LIKE)
    assert set(a.detections["Metric"]) == set(ECOM_LIKE)
    inc = incident_on(a, 100)
    assert inc["Title"] == "Sales drop" and inc["Category"] == "metric_drop"
    assert inc["Metric_Directions"]["Sales"] == "drop" and inc["Impact"] == "neutral"
    assert a.roles["Sales"] == "Financial (revenue)" and a.roles["Marketing_Spend"] == "Spend"


def test_different_domain_works_and_keeps_negative_values():
    df = upload_frame(WEATHER, seasonal=False, noise=0.02)
    df["Temperature"] = df["Temperature"] - 30  # below zero is legitimate
    df.loc[110, "Temperature"] += 25
    a = analyze_upload(df, "Date", list(WEATHER))
    assert (a.cleaned["Temperature"] < 0).any()
    assert a.cleaning.negative_values_clipped == 0
    assert any("negative" in w and "kept as-is" in w for w in a.validation.warnings)
    inc = incident_on(a, 110)
    assert inc["Title"] == "Temperature spike"
    assert "Revenue" not in " ".join(a.incidents["Evidence"])


def test_only_selected_metrics_are_analyzed_and_date_column_can_have_any_name():
    df = upload_frame(ECOM_LIKE, date_name="order_day")
    a = analyze_upload(df, "order_day", ["Sales", "Returns"])
    assert a.metrics == ["Sales", "Returns"] and set(a.detections["Metric"]) == {"Sales", "Returns"}
    assert "Date" in a.cleaned.columns and "Customers" not in a.cleaned.columns


def test_multi_metric_incident_gets_a_multi_metric_title_and_evidence_cells():
    df = upload_frame(ECOM_LIKE)
    for col in ("Sales", "Customers"):
        df.loc[100, col] *= 0.4
    a = analyze_upload(df, "Date", list(ECOM_LIKE))
    inc = incident_on(a, 100)
    assert inc["Category"] == "multi_metric_shift" and inc["Title"].startswith("Customers and 1 other metric")
    cells = incident_cells(inc, a.detections)
    assert list(cells.columns) == ["Date", "Metric", "Value", "Baseline", "Z_Score", "Status"]
    assert {"Sales", "Customers"} <= set(cells["Metric"])
    assert (cells["Status"] == "anomalous (below baseline)").all()


# ---- messy uploads ----------------------------------------------------------------------------
def test_messy_upload_runs_and_reports_what_it_did():
    df = upload_frame({"Sales": 12000, "Customers": 450})
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Sales"] = df["Sales"].map(lambda v: f"${v:,.2f}")  # numbers stored as text
    df.loc[7, "Date"] = "garbage"
    df.loc[9, "Customers"] = None
    df = pd.concat([df.drop(index=[30]), df.iloc[[3]]], ignore_index=True)
    a = analyze_upload(df, "Date", ["Sales", "Customers"])
    assert any("invalid" in n for n in a.notes)
    assert a.cleaning.duplicate_rows_dropped == 1 and a.cleaning.missing_days_added >= 1
    assert a.cleaned[["Sales", "Customers"]].notna().all().all()


def test_stringy_numbers_are_converted_and_counted():
    df = upload_frame({"Sales": 12000})
    df["Sales"] = df["Sales"].round().astype(int).astype(str)
    df.loc[4, "Sales"] = "n/a"
    n = normalize_dataset(df, "Date", ["Sales"])
    assert n.non_numeric_blanked == 1 and any("not numeric" in x for x in n.notes)
    assert n.frame["Sales"].isna().sum() == 1


# ---- aggregation ---------------------------------------------------------------------------------
def multi_row_frame():
    base = upload_frame({"Sales": 1000, "Temperature": 20.0}, n=60, seasonal=False, noise=0.0)
    rows = []
    for store in range(3):
        part = base.copy()
        part["Store"] = store
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def test_several_rows_per_date_are_aggregated_with_role_based_functions():
    n = normalize_dataset(multi_row_frame(), "Date", ["Sales", "Temperature"])
    assert len(n.frame) == 60
    assert n.aggregation == {"Sales": "sum", "Temperature": "mean"}
    assert n.frame["Sales"].iloc[0] == pytest.approx(3000) and n.frame["Temperature"].iloc[0] == pytest.approx(20)
    assert any("aggregated" in x for x in n.notes)


def test_aggregation_can_be_overridden():
    n = normalize_dataset(multi_row_frame(), "Date", ["Sales"], aggregation="mean")
    assert n.frame["Sales"].iloc[0] == pytest.approx(1000) and n.aggregation == {"Sales": "mean"}


def test_hourly_data_is_aggregated_to_days():
    hourly = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=24 * 80, freq="h"), "Orders": 5.0})
    a = analyze_upload(hourly, "ts", ["Orders"])
    assert len(a.cleaned) == 80 and (a.cleaned["Orders"] == 120).all()


# ---- unsuitable datasets ----------------------------------------------------------------------------
def test_weekly_data_raises_a_friendly_error():
    weekly = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=60, freq="7D"), "Sales": 1.0})
    with pytest.raises(UnsupportedDatasetError, match="weekly"):
        analyze_upload(weekly, "Date", ["Sales"])


def test_bad_selections_raise_friendly_errors():
    df = upload_frame({"Sales": 1000})
    with pytest.raises(UnsupportedDatasetError, match="Select at least one"):
        analyze_upload(df, "Date", [])
    with pytest.raises(UnsupportedDatasetError, match="not found"):
        analyze_upload(df, "Nope", ["Sales"])
    with pytest.raises(UnsupportedDatasetError, match="not found"):
        analyze_upload(df, "Date", ["Missing_Column"])


def test_a_metric_literally_named_date_does_not_collide():
    df = upload_frame({"Sales": 1000}, date_name="Timestamp")
    df["Date"] = np.random.default_rng(0).normal(50, 5, len(df))
    a = analyze_upload(df, "Timestamp", ["Sales", "Date"])
    assert a.metrics == ["Sales", "Date (metric)"]


def test_business_day_data_notes_the_filled_days():
    df = upload_frame({"Sales": 1000})
    df = df[df["Date"].dt.dayofweek < 5]
    a = analyze_upload(df, "Date", ["Sales"])
    assert any("calendar day" in n and "carried forward" in n for n in a.notes)


def test_constant_metric_and_short_history_are_reported():
    df = upload_frame({"Sales": 1000, "Flat": 5.0}, n=20, noise=0.0)
    df["Flat"] = 5.0
    a = analyze_upload(df, "Date", ["Sales", "Flat"])
    assert any("Flat" in n and "constant" in n for n in a.notes)
    assert any("Only 20 day" in n for n in a.notes)
    assert a.incidents.empty


# ---- demo path unchanged -------------------------------------------------------------------------------
def test_builtin_schema_analysis_is_unchanged():
    a = analyze(synthetic_kpis(conversion_drop_day=100))
    assert not a.generic and a.metrics == ALL_METRICS and a.roles is None and a.notes == []
    assert (a.incidents["Category"] == "conversion_drop").any() and "Title" not in a.incidents.columns


# ---- adapted modules ---------------------------------------------------------------------------------------
def test_validate_and_clean_accept_custom_metrics_and_can_keep_negatives():
    df = upload_frame({"Temp": 10.0}, n=80)
    df.loc[5, "Temp"] = -4.0
    with pytest.raises(SchemaError):
        validate_schema(df, metrics=["Temp", "Missing"])
    assert any("kept as-is" in w for w in validate_schema(df, metrics=["Temp"], negatives_are_errors=False).warnings)
    kept, report = clean_data(df, metrics=["Temp"], clip_negatives=False)
    assert kept.loc[5, "Temp"] == -4.0 and report.negative_values_clipped == 0
    clipped, report = clean_data(df, metrics=["Temp"])
    assert clipped.loc[5, "Temp"] == 0 and report.negative_values_clipped == 1


def test_generic_rules_describe_what_moved():
    def classify(directions):
        inc = pd.DataFrame([{"Incident_ID": 1, "Metric_Directions": directions}])
        return classify_incidents(inc, generic=True).iloc[0]

    assert classify({"Sales": "spike"})["Category"] == "metric_spike"
    assert classify({"Sales": "mixed"})["Category"] == "metric_swing"
    r = classify({"Sales": "drop", "Returns": "spike", "Visits": "drop"})
    assert r["Category"] == "multi_metric_shift" and r["Title"] == "Returns and 2 other metrics"
    assert "does not establish a cause" in r["Pattern"]
    # The built-in rules are untouched for the demo schema.
    demo = classify_incidents(pd.DataFrame([{"Incident_ID": 1, "Metric_Directions": {"Refunds": "spike"}}]))
    assert demo.iloc[0]["Category"] == "refund_issue"


def test_severity_revenue_metrics_and_rescaling():
    inc = pd.DataFrame([{
        "Incident_ID": 1, "Peak_Abs_Z": 10.0, "Duration_Days": 5, "Metrics": ["A", "B", "C", "D"],
        "Metric_Directions": {"A": "drop", "B": "drop", "C": "drop", "D": "drop"}, "Impact": "neutral",
    }])
    none = score_incidents(inc, revenue_metrics=(), derived_metrics=()).iloc[0]
    with_revenue = score_incidents(inc.assign(Metric_Directions=[{"A": "drop", "B": "drop", "C": "drop", "D": "drop"}]),
                                   revenue_metrics=("A",), derived_metrics=()).iloc[0]
    assert none["Severity_Score"] == 100.0 and with_revenue["Severity_Score"] == 100.0  # both can reach the top
    assert "rescaled" in none["Severity_Breakdown"]
    weak = inc.assign(Peak_Abs_Z=5.0, Duration_Days=1, Metrics=[["A", "B"]], Metric_Directions=[{"A": "drop", "B": "drop"}])
    assert score_incidents(weak, revenue_metrics=(), derived_metrics=()).iloc[0]["Severity_Score"] \
        > score_incidents(weak, revenue_metrics=("Z",), derived_metrics=()).iloc[0]["Severity_Score"]


def test_derived_names_only_count_as_derived_for_the_builtin_schema():
    inc = pd.DataFrame([{"Incident_ID": 1, "Peak_Abs_Z": 6.0, "Duration_Days": 1, "Metrics": ["Refund_Rate", "Orders"],
                         "Metric_Directions": {"Refund_Rate": "spike", "Orders": "drop"}, "Impact": "neutral"}])
    builtin = score_incidents(inc).iloc[0]["Severity_Breakdown"]
    upload = score_incidents(inc, revenue_metrics=(), derived_metrics=()).iloc[0]["Severity_Breakdown"]
    assert "weak single-metric flag" in builtin and "weak single-metric flag" not in upload


# ---- AI layer on arbitrary metrics -----------------------------------------------------------------------------
def weather_analysis():
    df = upload_frame(WEATHER, seasonal=False, noise=0.02)
    df.loc[110, "Temperature"] += 25
    return analyze_upload(df, "Date", list(WEATHER))


class FakeClient:
    def __init__(self, response):
        self.response = response

    def complete(self, system, user):
        self.user = user
        return json.dumps(self.response)


def test_facts_carry_metric_types_and_template_passes_validation():
    a = weather_analysis()
    inc = incident_on(a, 110)
    facts = build_incident_facts(inc, a.detections, a.roles)
    assert facts["metrics"][0]["metric"] == "Temperature" and facts["metrics"][0]["metric_type"] == "Generic numeric metric"
    assert validate_note(template_note(facts), facts, a.metrics) == []
    assert "Humidity" not in json.dumps(facts) and "Revenue" not in json.dumps(facts)


def test_validator_rejects_other_dataset_metrics_but_not_the_incidents_own():
    a = weather_analysis()
    inc = incident_on(a, 110)
    facts = build_incident_facts(inc, a.detections, a.roles)
    own = {"summary": f"Temperature was {facts['metrics'][0]['pct_vs_same_weekday_baseline']}% above its same-weekday baseline.", "checks": []}
    other = {"summary": "Temperature jumped while Humidity stayed normal.", "checks": []}
    assert validate_note(own, facts, a.metrics) == []
    assert any("Humidity" in p for p in validate_note(other, facts, a.metrics))
    # without the dataset's metric list the check cannot know about Humidity
    assert validate_note(other, facts) == []


def test_ai_summary_flow_on_uploaded_data_never_sees_the_csv():
    a = weather_analysis()
    inc = incident_on(a, 110)
    facts = build_incident_facts(inc, a.detections, a.roles)
    client = FakeClient({"summary": f"Temperature was {facts['metrics'][0]['pct_vs_same_weekday_baseline']}% above baseline.", "checks": ["Check the sensor"]})
    note = summarize_incident(facts, client, a.metrics)
    assert note["source"] == "llm"
    sent = client.user
    assert "Temperature" in sent and "Pressure" not in sent  # only the incident's facts
    assert len(sent) < 2500  # a fact sheet, not the dataset
    out = summarize_incidents(a.incidents, a.detections, None, a.metrics, a.roles)
    assert set(out["Summary_Source"]) == {"template"} and out["Summary"].notna().all()


def test_stated_cause_is_still_rejected_for_uploaded_data():
    a = weather_analysis()
    facts = build_incident_facts(incident_on(a, 110), a.detections, a.roles)
    note = summarize_incident(facts, FakeClient({"summary": "Temperature rose because of a heatwave.", "checks": []}), a.metrics)
    assert note["source"] == "template_fallback"
