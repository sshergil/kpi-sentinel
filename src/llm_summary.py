"""LLM summarization layer: explain incidents in business language, safely.

Design rules (the project's core principle):
  * Statistics decide what is anomalous. The LLM only explains incidents
    the detector has already flagged, using a small structured fact sheet.
  * Every LLM answer is machine-checked before use. If it contains a number
    that is not in the facts, states a cause as fact, or mentions a metric
    that is not part of the incident, it is discarded and a deterministic
    template summary is used instead. Unverified text is never shown.
  * Without an API key (or if the call fails) the system still works: the
    template summary is used. This also makes the dashboard demo-able for free.

The LLM client is any object with `complete(system, user) -> str`, so tests
(and other providers) can plug in without network access.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

import pandas as pd

from src.derived_metrics import ALL_METRICS

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 400
MAX_SUMMARY_WORDS = 80
MAX_CHECKS = 3

SYSTEM_PROMPT = (
    "You write short incident notes for an e-commerce analytics team. A "
    "statistical detector has already decided that the metrics below are "
    "anomalous; you only explain what happened. Rules: use ONLY the facts "
    "provided; copy numbers exactly as given (never round or invent any); "
    "never state or imply a cause as fact, use hedged wording such as "
    "'consistent with'; do not mention metrics that are not in the facts; "
    f"keep the summary under {MAX_SUMMARY_WORDS} words. Return ONLY JSON: "
    '{"summary": "...", "checks": ["...", "..."]} where checks are up to '
    f"{MAX_CHECKS} short, concrete things the team could look into."
)

CAUSAL_PATTERNS = [
    r"\bcaused? by\b", r"\bdue to\b", r"\bbecause\b", r"\bas a result of\b",
    r"\bresulted from\b", r"\bled to\b", r"\btriggered by\b", r"\bis responsible\b",
]

DEFAULT_CHECKS = {
    "demand_spike": ["Confirm whether a promotion, campaign or press mention ran on these dates",
                     "Check that inventory and fulfilment kept up with the extra orders"],
    "traffic_quality": ["Break the traffic spike down by source and look for bot or referral patterns",
                        "Compare engagement and conversion for the extra traffic"],
    "conversion_drop": ["Review checkout and payment error rates for these dates",
                        "Check for site releases or pricing changes around the start date"],
    "traffic_drop": ["Check site uptime and error logs for these dates",
                     "Review search ranking and campaign delivery for the affected period"],
    "marketing_efficiency": ["Review campaign budgets and bids that changed around these dates",
                             "Compare spend and revenue by channel"],
    "refund_issue": ["Group refunds by product, batch and carrier",
                     "Review customer support tickets from the same period"],
    "aov_spike": ["Look for a small number of unusually large orders",
                  "Check for pricing, bundling or promotion changes"],
    "unclassified": ["Review the flagged metrics for these dates in the raw data"],
}


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------

def build_incident_facts(incident: pd.Series, detections: pd.DataFrame) -> dict:
    """Build the structured fact sheet the LLM (or template) may draw on."""
    start, end = incident["Start_Date"], incident["End_Date"]
    cells = detections[
        detections["Is_Anomaly"]
        & detections["Date"].between(start, end)
        & detections["Metric"].isin(incident["Metrics"])
    ]
    metrics = []
    for metric, g in cells.groupby("Metric"):
        base = g["Baseline_Mean"]
        pct = ((g["Value"] - base) / base.where(base != 0) * 100).mean()
        metrics.append({
            "metric": metric,
            "direction": incident["Metric_Directions"][metric],
            "days_flagged": int(len(g)),
            "pct_vs_same_weekday_baseline": None if pd.isna(pct) else int(round(float(pct))),
            "peak_abs_z": round(float(g["Z_Score"].abs().max()), 1),
        })
    metrics.sort(key=lambda m: -m["peak_abs_z"])
    return {
        "incident_id": int(incident["Incident_ID"]),
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "duration_days": int(incident["Duration_Days"]),
        "category": incident["Category"],
        "impact": incident["Impact"],
        "pattern": incident["Pattern"],
        "severity_score": float(incident["Severity_Score"]),
        "severity_label": incident["Severity_Label"],
        "n_metrics": len(metrics),
        "metrics": metrics,
    }


# --------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------

def _metric_phrase(m: dict) -> str:
    pct = m["pct_vs_same_weekday_baseline"]
    if pct is None:
        return f"{m['metric']} was flagged ({m['direction']})"
    side = "above" if pct > 0 else "below"
    return f"{m['metric']} was {abs(pct)}% {side} its same-weekday baseline"


def template_note(facts: dict) -> dict:
    """Summary and checks built only from the facts; no LLM involved."""
    when = (facts["start_date"] if facts["start_date"] == facts["end_date"]
            else f"{facts['start_date']} to {facts['end_date']}")
    days = f"{facts['duration_days']} day" + ("" if facts["duration_days"] == 1 else "s")
    moves = "; ".join(_metric_phrase(m) for m in facts["metrics"])
    summary = (f"{when} ({days}), severity {facts['severity_label']} "
               f"({facts['severity_score']:g}): {moves}. {facts['pattern']}")
    return {"summary": summary, "checks": list(DEFAULT_CHECKS.get(facts["category"], DEFAULT_CHECKS["unclassified"]))}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _numbers(text: str) -> set[float]:
    return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}


def _mentioned_metrics(text: str) -> set[str]:
    t = text.lower()
    found = set()
    for name in sorted(ALL_METRICS, key=len, reverse=True):  # longest first
        for form in {name.lower(), name.lower().replace("_", " ")}:
            if form in t:
                found.add(name)
                t = t.replace(form, " ")
    return found


def validate_note(note: dict, facts: dict) -> list[str]:
    """Return a list of problems; empty means the note is safe to show."""
    problems: list[str] = []
    summary, checks = note.get("summary"), note.get("checks")
    if not isinstance(summary, str) or not summary.strip():
        return ["summary missing"]
    if not isinstance(checks, list) or not all(isinstance(c, str) for c in checks):
        return ["checks must be a list of strings"]
    if len(summary.split()) > MAX_SUMMARY_WORDS:
        problems.append("summary too long")
    if len(checks) > MAX_CHECKS:
        problems.append("too many checks")

    text = " ".join([summary] + checks)

    allowed_numbers = _numbers(json.dumps(facts))
    invented = sorted(_numbers(text) - allowed_numbers)
    if invented:
        problems.append(f"numbers not in facts: {', '.join(f'{n:g}' for n in invented)}")

    causal = [p for p in CAUSAL_PATTERNS if re.search(p, text, re.IGNORECASE)]
    if causal:
        problems.append("states a cause as fact")

    allowed_metrics = {m["metric"] for m in facts["metrics"]} | _mentioned_metrics(facts["pattern"])
    extra = sorted(_mentioned_metrics(text) - allowed_metrics)
    if extra:
        problems.append(f"mentions metrics outside the incident: {', '.join(extra)}")
    return problems


# --------------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def summarize_incident(facts: dict, client: LLMClient | None = None) -> dict:
    """Return {summary, checks, source, problems} for one incident.

    source is "llm" (verified LLM text), "template" (no client), or
    "template_fallback" (LLM output rejected or the call failed).
    """
    if client is None:
        return {**template_note(facts), "source": "template", "problems": []}
    try:
        raw = client.complete(SYSTEM_PROMPT, "Incident facts (JSON):\n" + json.dumps(facts, indent=2))
        note = _parse_json(raw)
        problems = validate_note(note, facts)
    except Exception as exc:  # network, auth, bad JSON: never break the pipeline
        return {**template_note(facts), "source": "template_fallback",
                "problems": [f"{type(exc).__name__}: {exc}"]}
    if problems:
        return {**template_note(facts), "source": "template_fallback", "problems": problems}
    return {"summary": note["summary"].strip(), "checks": note["checks"], "source": "llm", "problems": []}


def summarize_incidents(
    incidents: pd.DataFrame, detections: pd.DataFrame, client: LLMClient | None = None
) -> pd.DataFrame:
    """Return a copy of `incidents` with Summary, Checks, Summary_Source columns."""
    out = incidents.copy()
    notes = [summarize_incident(build_incident_facts(row, detections), client)
             for _, row in out.iterrows()]
    out["Summary"] = [n["summary"] for n in notes]
    out["Checks"] = [n["checks"] for n in notes]
    out["Summary_Source"] = [n["source"] for n in notes]
    return out


# --------------------------------------------------------------------------
# Anthropic client
# --------------------------------------------------------------------------

class AnthropicClient:
    """Thin wrapper around the Anthropic Python SDK (`pip install anthropic`)."""

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = MAX_TOKENS):
        import anthropic  # imported lazily: only needed when the LLM is actually used

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model or os.environ.get("KPI_SENTINEL_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


def get_default_client() -> LLMClient | None:
    """An AnthropicClient if ANTHROPIC_API_KEY is set and the SDK is installed, else None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicClient()
    except ImportError:
        return None
