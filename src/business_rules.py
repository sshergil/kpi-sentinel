"""Cross-metric business rules: label incidents by the pattern they show.

Rules look at which metrics moved in which direction within an incident
and assign the first matching category. Descriptions use hedged wording
("consistent with ...") because a statistical pattern does not prove a
cause. Rules never decide *whether* something is anomalous; that was
already decided by the detector.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import pandas as pd


class Rule(NamedTuple):
    category: str
    impact: str  # "positive", "negative" or "neutral" for the business
    description: str
    matches: Callable[[dict], bool]


def _up(d: dict, metric: str) -> bool:
    return d.get(metric) == "spike"


def _down(d: dict, metric: str) -> bool:
    return d.get(metric) == "drop"


# Order matters: the first matching rule wins.
RULES = [
    Rule(
        "demand_spike", "positive",
        "Traffic, orders and revenue rose together, consistent with a genuine "
        "demand increase such as a promotion or viral moment.",
        lambda d: _up(d, "Website_Traffic") and _up(d, "Orders") and _up(d, "Revenue"),
    ),
    Rule(
        "marketing_efficiency", "negative",
        "Ad spend jumped without a proportional revenue gain, consistent with "
        "declining marketing efficiency.",
        lambda d: _up(d, "Ad_Spend")
        and (_down(d, "Revenue_per_Ad_Dollar") or not _up(d, "Revenue")),
    ),
    Rule(
        "traffic_quality", "negative",
        "Traffic rose without a matching rise in orders or revenue, consistent "
        "with low-quality or non-converting traffic rather than real demand.",
        lambda d: _up(d, "Website_Traffic") and not _up(d, "Orders") and not _up(d, "Revenue"),
    ),
    Rule(
        "conversion_drop", "negative",
        "Conversion and sales fell while traffic did not, consistent with a "
        "checkout or purchasing-behaviour problem rather than a traffic problem.",
        lambda d: _down(d, "Conversion_Rate")
        and (_down(d, "Orders") or _down(d, "Revenue"))
        and not _down(d, "Website_Traffic"),
    ),
    Rule(
        "traffic_drop", "negative",
        "Traffic fell and sales fell with it, consistent with a problem upstream "
        "of the purchase funnel such as an outage or lost acquisition.",
        lambda d: _down(d, "Website_Traffic") and (_down(d, "Orders") or _down(d, "Revenue")),
    ),
    Rule(
        "refund_issue", "negative",
        "Refunds rose abnormally, consistent with a product-quality or "
        "fulfilment problem.",
        lambda d: _up(d, "Refunds") or (_up(d, "Refund_Rate") and not _down(d, "Revenue")),
    ),
    Rule(
        "aov_spike", "positive",
        "Average order value jumped without more orders, consistent with a few "
        "unusually large purchases.",
        lambda d: _up(d, "Average_Order_Value") and not _up(d, "Orders"),
    ),
]

UNCLASSIFIED = Rule(
    "unclassified", "neutral",
    "Unusual movement that does not match a known cross-metric pattern.",
    lambda d: True,
)


def classify_incidents(incidents: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `incidents` with Category, Impact, Pattern, Evidence."""
    out = incidents.copy()
    categories, impacts, patterns, evidence = [], [], [], []
    for directions in out["Metric_Directions"]:
        rule = next((r for r in RULES if r.matches(directions)), UNCLASSIFIED)
        categories.append(rule.category)
        impacts.append(rule.impact)
        patterns.append(rule.description)
        evidence.append(", ".join(f"{m} {directions[m]}" for m in sorted(directions)))
    out["Category"] = categories
    out["Impact"] = impacts
    out["Pattern"] = patterns
    out["Evidence"] = evidence
    return out
