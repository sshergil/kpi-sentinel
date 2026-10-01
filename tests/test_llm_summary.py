import json
import os
import sys
import types

import pandas as pd

from src.business_rules import RULES
from src.llm_summary import (
    DEFAULT_CHECKS,
    AnthropicClient,
    build_incident_facts,
    get_default_client,
    summarize_incident,
    summarize_incidents,
    template_note,
    validate_note,
)

DAY = pd.Timestamp("2026-04-16")


def detections():
    rows = [
        {"Date": DAY, "Metric": "Revenue", "Value": 11800.0, "Baseline_Mean": 20000.0,
         "Z_Score": -6.2, "Is_Anomaly": True},
        {"Date": DAY, "Metric": "Orders", "Value": 240.0, "Baseline_Mean": 400.0,
         "Z_Score": -5.1, "Is_Anomaly": True},
        {"Date": DAY, "Metric": "Refunds", "Value": 600.0, "Baseline_Mean": 600.0,
         "Z_Score": 0.1, "Is_Anomaly": False},  # not part of the incident
        {"Date": DAY - pd.Timedelta(days=7), "Metric": "Revenue", "Value": 20000.0,
         "Baseline_Mean": 20000.0, "Z_Score": 0.0, "Is_Anomaly": False},
    ]
    return pd.DataFrame(rows)


def incident(pattern="Conversion and sales fell, consistent with a checkout problem.",
             category="conversion_drop"):
    return pd.Series({
        "Incident_ID": 7, "Start_Date": DAY, "End_Date": DAY, "Duration_Days": 1,
        "Metrics": ["Orders", "Revenue"],
        "Metric_Directions": {"Orders": "drop", "Revenue": "drop"},
        "Category": category, "Impact": "negative", "Pattern": pattern,
        "Severity_Score": 84.8, "Severity_Label": "Critical",
    })


def facts(**kw):
    return build_incident_facts(incident(**kw), detections())


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


GOOD = {
    "summary": "On 2026-04-16 Revenue was 41% below its same-weekday baseline and Orders "
               "40% below, consistent with a conversion problem.",
    "checks": ["Review checkout error rates for 2026-04-16"],
}


# --- facts -----------------------------------------------------------------

def test_facts_contain_only_flagged_incident_metrics_with_numbers():
    f = facts()
    assert f["start_date"] == "2026-04-16" and f["duration_days"] == 1
    assert [m["metric"] for m in f["metrics"]] == ["Revenue", "Orders"]  # strongest z first
    revenue = f["metrics"][0]
    assert revenue["pct_vs_same_weekday_baseline"] == -41
    assert revenue["peak_abs_z"] == 6.2 and revenue["direction"] == "drop"
    assert f["n_metrics"] == 2
    assert "Refunds" not in json.dumps(f)


def test_facts_are_json_serializable():
    json.dumps(facts())


# --- template ----------------------------------------------------------------

def test_no_client_uses_template():
    n = summarize_incident(facts())
    assert n["source"] == "template" and n["problems"] == []
    assert "41%" in n["summary"] and "2026-04-16" in n["summary"]
    assert n["checks"] == DEFAULT_CHECKS["conversion_drop"]


def test_template_output_passes_its_own_validation_for_every_category():
    for rule in RULES:
        f = facts(pattern=rule.description, category=rule.category)
        assert validate_note(template_note(f), f) == [], rule.category


# --- accepting / rejecting LLM output ----------------------------------------

def test_valid_llm_output_is_used():
    client = FakeClient(GOOD)
    n = summarize_incident(facts(), client)
    assert n["source"] == "llm" and n["problems"] == []
    assert n["summary"] == GOOD["summary"]
    system, user = client.calls[0]
    assert "ONLY the facts" in system and '"pct_vs_same_weekday_baseline": -41' in user


def test_json_in_code_fence_is_accepted():
    n = summarize_incident(facts(), FakeClient("```json\n" + json.dumps(GOOD) + "\n```"))
    assert n["source"] == "llm"


def test_invented_number_is_rejected():
    bad = {**GOOD, "summary": "Revenue was 57% below baseline, consistent with a conversion problem."}
    n = summarize_incident(facts(), FakeClient(bad))
    assert n["source"] == "template_fallback"
    assert any("57" in p for p in n["problems"])
    assert "57%" not in n["summary"]


def test_stated_cause_is_rejected():
    bad = {**GOOD, "summary": "Revenue fell 41% because of a broken checkout."}
    n = summarize_incident(facts(), FakeClient(bad))
    assert n["source"] == "template_fallback" and "states a cause as fact" in n["problems"]


def test_cause_in_checks_is_rejected_too():
    bad = {**GOOD, "checks": ["Investigate the drop due to the new release"]}
    assert summarize_incident(facts(), FakeClient(bad))["source"] == "template_fallback"


def test_metric_outside_the_incident_is_rejected():
    bad = {**GOOD, "summary": "Revenue was 41% below baseline while Ad_Spend also rose."}
    n = summarize_incident(facts(), FakeClient(bad))
    assert n["source"] == "template_fallback"
    assert any("Ad_Spend" in p for p in n["problems"])


def test_metric_named_in_the_rule_pattern_may_be_mentioned():
    f = facts(pattern="Website Traffic held steady, consistent with a checkout problem.")
    ok = {**GOOD, "summary": "Revenue was 41% below baseline while website traffic held steady."}
    assert summarize_incident(f, FakeClient(ok))["source"] == "llm"


def test_longer_metric_names_are_not_confused_with_shorter_ones():
    # "revenue per ad dollar" must not count as a mention of Revenue/Ad_Spend.
    f = facts()
    text = {**GOOD, "summary": "Revenue was 41% below baseline; revenue per ad dollar is not in the facts."}
    problems = validate_note(text, f)
    assert any("Revenue_per_Ad_Dollar" in p for p in problems)
    assert not any("Revenue," in p or p.endswith("Revenue") for p in problems)


def test_overlong_summary_is_rejected():
    bad = {**GOOD, "summary": " ".join(["word"] * 120)}
    assert summarize_incident(facts(), FakeClient(bad))["source"] == "template_fallback"


def test_malformed_output_and_client_errors_fall_back_without_raising():
    for client in (FakeClient("not json"), FakeClient('["a list"]'),
                   FakeClient({"summary": "x"}), FakeClient(error=RuntimeError("boom"))):
        n = summarize_incident(facts(), client)
        assert n["source"] == "template_fallback" and n["problems"]
        assert n["summary"] and n["checks"]


def test_runtime_error_is_reported():
    n = summarize_incident(facts(), FakeClient(error=RuntimeError("boom")))
    assert any("RuntimeError" in p for p in n["problems"])


# --- dataframe helper ---------------------------------------------------------

def test_summarize_incidents_adds_columns_and_keeps_input():
    inc = pd.DataFrame([incident()])
    before = inc.copy()
    out = summarize_incidents(inc, detections(), FakeClient(GOOD))
    assert list(out["Summary_Source"]) == ["llm"]
    assert out.loc[0, "Summary"] == GOOD["summary"]
    assert out.loc[0, "Checks"] == GOOD["checks"]
    pd.testing.assert_frame_equal(inc, before)


# --- Anthropic wrapper ---------------------------------------------------------

def test_anthropic_client_calls_messages_api_and_joins_text_blocks():
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = lambda t, text: types.SimpleNamespace(type=t, text=text)  # noqa: E731
            return types.SimpleNamespace(content=[block("text", '{"a":'), block("tool_use", "x"), block("text", " 1}")])

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = FakeAnthropic
    original = sys.modules.get("anthropic")
    sys.modules["anthropic"] = fake_module
    try:
        client = AnthropicClient(model="test-model")
        assert client.complete("SYS", "USER") == '{"a": 1}'
    finally:
        if original is None:
            del sys.modules["anthropic"]
        else:
            sys.modules["anthropic"] = original
    assert captured["model"] == "test-model" and captured["system"] == "SYS"
    assert captured["messages"] == [{"role": "user", "content": "USER"}]
    assert captured["max_tokens"] > 0


def test_default_client_is_none_without_api_key():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        assert get_default_client() is None
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
