"""Measure how often LLM summaries survive automatic fact-checking.

This is an automated factuality check, not a human judgement of quality: it
counts how many model answers pass the same validator used in production
(numbers grounded in the facts, no stated causes, no out-of-scope metrics).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.llm_summary import LLMClient, build_incident_facts, summarize_incident


def _problem_type(problem: str) -> str:
    if problem.startswith("numbers not in facts"):
        return "invented_number"
    if problem.startswith("states a cause"):
        return "stated_cause"
    if problem.startswith("mentions metrics"):
        return "out_of_scope_metric"
    if problem.startswith(("summary too long", "too many checks")):
        return "too_long"
    if ": " in problem and problem.split(":")[0].endswith(("Error", "Exception")):
        return "call_or_parse_error"
    return "malformed_output"


def evaluate_summaries(incidents: pd.DataFrame, detections: pd.DataFrame, client: LLMClient) -> dict:
    """Summarize every incident with `client` and report acceptance statistics."""
    sources, problems = [], Counter()
    for _, row in incidents.iterrows():
        note = summarize_incident(build_incident_facts(row, detections), client)
        sources.append(note["source"])
        problems.update(_problem_type(p) for p in note["problems"])
    n = len(sources)
    accepted = sources.count("llm")
    return {
        "n_incidents": n,
        "llm_accepted": accepted,
        "fallbacks": n - accepted,
        "acceptance_rate": accepted / n if n else 0.0,
        "problem_types": dict(problems),
    }
