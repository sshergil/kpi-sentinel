"""Review an uploaded CSV before analysis: what we understood, what we will change, what blocks.

Nothing here modifies data or decides what is anomalous. It reads the raw table and the
user's current column choices and returns plain-English findings for the confirmation step:

  * a column-mapping table (what each column was understood to be, and why)
  * notes that explain ambiguous choices (two possible time columns, day-first dates, ...)
  * data-quality issues, each saying what was found and what will happen to it
  * a verdict: ready, ready with cautions, or blocked with the reason

The same functions are used by the Streamlit app and by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.baselines import MIN_WEEKS
from src.data_loader import MIN_ROWS, NUMERIC_COLUMNS
from src.dataset_profiler import (
    ROLE_LABELS, DatasetProfile, TimeAxis, classify_role, date_order_ambiguous, describe_time_axis,
    metric_notes, parse_dates, profile_dataset, suggest_aggregation, to_numeric, tokens, uses_decimal_comma,
)

MIN_DAYS = MIN_WEEKS * 7 + 1  # first day the detector can judge: 4 prior same-weekdays
MISSING_WARN = 0.05
MISSING_STRONG = 0.20
MISSING_DAYS_WARN = 0.05

# Names of the built-in e-commerce metrics a user's column may correspond to (token-joined).
BUILTIN_ALIASES: dict[str, set[str]] = {
    "Revenue": {"revenue", "total_revenue", "sales", "total_sales", "net_sales", "gross_sales", "net_revenue"},
    "Orders": {"orders", "order_count", "num_orders", "total_orders", "number_of_orders"},
    "Website_Traffic": {"website_traffic", "traffic", "sessions", "visits", "visitors", "site_visits", "web_traffic"},
    "Conversion_Rate": {"conversion_rate", "conversion", "cvr", "conv_rate", "conversion_pct"},
    "Ad_Spend": {"ad_spend", "adspend", "marketing_spend", "ad_cost", "advertising_spend", "ad_budget", "marketing_cost"},
    "Refunds": {"refunds", "refund_amount", "total_refunds", "refunded_amount", "refund_value"},
    "Average_Order_Value": {"average_order_value", "aov", "avg_order_value", "avg_order_val"},
}
# How each built-in metric combines when a file has several rows per day.
BUILTIN_AGGREGATION = {"Revenue": "sum", "Orders": "sum", "Website_Traffic": "sum", "Conversion_Rate": "mean",
                       "Ad_Spend": "sum", "Refunds": "sum", "Average_Order_Value": "mean"}
assert set(BUILTIN_ALIASES) == set(NUMERIC_COLUMNS) == set(BUILTIN_AGGREGATION)


@dataclass
class QualityIssue:
    level: str  # "info" | "warning" | "blocking"
    title: str
    detail: str  # what was found
    action: str = ""  # what will happen to it, or what the user can do


@dataclass
class UploadReview:
    profile: DatasetProfile
    date_column: str | None
    metrics: list[str]
    axis: TimeAxis | None
    issues: list[QualityIssue] = field(default_factory=list)
    mapping_notes: list[str] = field(default_factory=list)
    columns: pd.DataFrame = field(default_factory=pd.DataFrame)
    builtin_mapping: dict[str, str] | None = None  # built-in metric -> the user's column
    duplicate_rows: int = 0
    recommend_drop_duplicates: bool = False

    @property
    def blocking(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.level == "blocking"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def status(self) -> str:
        """'blocked' (cannot analyze), 'caution' (can, with caveats) or 'ready'."""
        return "blocked" if self.blocking else "caution" if self.warnings else "ready"

    @property
    def headline(self) -> str:
        if self.blocking:
            return self.blocking[0].title
        if self.warnings:
            n = len(self.warnings)
            return f"Ready to analyze, with {n} thing{'s' if n > 1 else ''} to be aware of"
        return "Ready to analyze"


# --------------------------------------------------------------------------
# Column-name mapping
# --------------------------------------------------------------------------

def normalize_column_name(name: str) -> str:
    """'Sales ($)' -> 'sales', 'Order Date' -> 'order_date', 'adSpend' -> 'ad_spend'."""
    return "_".join(tokens(name))


def suggest_builtin_mapping(columns: list[str]) -> dict[str, str] | None:
    """Map the user's columns onto the built-in e-commerce metrics, or None.

    Only returns a mapping when every one of the seven metrics has exactly one matching
    column. A partial or ambiguous match would silently change what the cross-metric rules
    mean, so it is not attempted.
    """
    by_name: dict[str, list[str]] = {}
    for c in columns:
        by_name.setdefault(normalize_column_name(c), []).append(c)
    mapping: dict[str, str] = {}
    for metric, aliases in BUILTIN_ALIASES.items():
        hits = [c for alias in aliases for c in by_name.get(alias, [])]
        if len(hits) != 1:
            return None
        mapping[metric] = hits[0]
    return mapping if len(set(mapping.values())) == len(mapping) else None


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------

def _examples(series: pd.Series, n: int = 3) -> str:
    return ", ".join(str(v) for v in series.dropna().astype(str).drop_duplicates().head(n))


def _pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.01 else "<1%"


def review_upload(
    raw: pd.DataFrame,
    date_column: str | None = None,
    metrics: list[str] | None = None,
    profile: DatasetProfile | None = None,
    overrides: dict[str, str] | None = None,
) -> UploadReview:
    """Build the confirmation-step review for `raw` with the current (or suggested) choices."""
    profile = profile or profile_dataset(raw)
    date_column = date_column if date_column is not None else profile.date_column
    metrics = [m for m in (metrics if metrics is not None else profile.metric_columns)
               if m in raw.columns and m != date_column]
    overrides = overrides or {}
    review = UploadReview(profile=profile, date_column=date_column, metrics=metrics, axis=None)
    add = review.issues.append

    review.duplicate_rows = int(raw.duplicated().sum())
    has_id = any(c.kind == "identifier" for c in profile.columns)
    review.builtin_mapping = suggest_builtin_mapping(list(raw.columns))

    bad_lines = int(raw.attrs.get("bad_lines", 0))
    if bad_lines:
        add(QualityIssue("warning", f"{bad_lines} malformed line(s) in the file were skipped",
                         "These lines had a different number of fields than the header (often an unquoted comma inside a value).",
                         "They are not part of the analysis. Fix them in the source file if they matter."))

    # ---- time column ------------------------------------------------------------------
    if date_column is None:
        add(QualityIssue("blocking", "No time column selected",
                         profile.problem or "A time column is required for time-series anomaly detection.",
                         "Choose the column that holds the date or timestamp of each row."))
    else:
        review.axis = axis = describe_time_axis(raw, date_column)
        if axis.n_invalid:
            bad = raw.loc[parse_dates(raw[date_column]).isna(), date_column]
            add(QualityIssue("warning", f"{axis.n_invalid} row(s) have a missing or unreadable date",
                             "Examples: " + (_examples(bad) or "(blank)") + ".",
                             "These rows cannot be placed on a timeline, so they are left out of the analysis."))
        if date_order_ambiguous(raw[date_column]):
            day_first = bool((parse_dates(raw[date_column]).dt.day <= 12).all())
            add(QualityIssue("warning", "Dates could be read day-first or month-first",
                             f"Every date in '{date_column}' is valid either way (for example {_examples(raw[date_column], 1)}).",
                             "We chose the reading that gives a continuous calendar" + (" (day-first)." if day_first else ".")
                             + " If the dates look wrong in the preview, re-export them as YYYY-MM-DD."))
        if axis.problem:
            add(QualityIssue("blocking", "This data is not suitable for daily anomaly detection", axis.problem,
                             "Upload daily data, or data with several rows per day (such as individual orders or hourly readings)."))
        else:
            span = (axis.last_date - axis.first_date).days + 1
            if span < MIN_DAYS:
                add(QualityIssue("blocking", f"Only {span} days of history, too short to find anomalies",
                                 f"The detector compares each day with the same weekday over the previous weeks and needs "
                                 f"at least {MIN_WEEKS} of them, so the first day it can judge is day {MIN_DAYS}.",
                                 f"Upload at least {MIN_DAYS} days; {MIN_ROWS} or more gives a steadier baseline."))
            elif span < MIN_ROWS:
                add(QualityIssue("warning", f"Short history ({span} days)",
                                 f"At least {MIN_ROWS} days are recommended for a reliable baseline.",
                                 "Results will be noisier, especially for the first weeks."))
            if axis.multiple_rows_per_date:
                add(QualityIssue("info", f"About {axis.rows_per_date:g} rows per day will be combined into one",
                                 f"'{date_column}' repeats ({axis.n_valid} rows, {axis.n_unique_dates} distinct days).",
                                 "Amounts are added up; averages, prices and rates are averaged (see the metric table; you can change it)."))
            if axis.missing_day_share > 0:
                level = "warning" if axis.missing_day_share > MISSING_DAYS_WARN else "info"
                n_missing = round(axis.missing_day_share * span)
                add(QualityIssue(level, f"{n_missing} calendar day(s) have no data ({_pct(axis.missing_day_share)} of the period)",
                                 "Common with business-day data or gaps in collection.",
                                 "The previous day's value is carried forward so the series has one row per day. "
                                 "Days filled this way are not real observations."))

    # ---- duplicates --------------------------------------------------------------------
    review.recommend_drop_duplicates = bool(review.duplicate_rows and has_id)
    if review.duplicate_rows:
        multi = review.axis is not None and review.axis.multiple_rows_per_date
        if multi and not has_id:
            add(QualityIssue("warning", f"{review.duplicate_rows} row(s) are exact duplicates of another row",
                             "There is no ID column, so identical rows may be genuine separate records (two identical sales).",
                             "They are kept by default. Tick 'Remove exact duplicate rows' if they are copies."))
        elif multi:
            add(QualityIssue("warning", f"{review.duplicate_rows} row(s) are exact duplicates of another row",
                             "An ID column is present, so identical rows are almost certainly the same record entered twice.",
                             "They are removed before the analysis (you can untick this)."))
        else:
            add(QualityIssue("info", f"{review.duplicate_rows} exact duplicate row(s)",
                             "Identical rows for the same day.", "Duplicates are dropped; only the first copy is kept."))

    # ---- metrics -------------------------------------------------------------------------
    if not metrics and date_column is not None:
        add(QualityIssue("blocking", "No numeric metrics selected",
                         profile.problem or "At least one numeric measure is needed.",
                         "Pick the numeric columns you want monitored."))
    numeric = {m: to_numeric(raw[m]) for m in metrics}
    for m in metrics:
        col, num = raw[m], numeric[m]
        n_missing = int(col.isna().sum())
        n_text = int((num.isna() & col.notna()).sum())
        n_neg = int((num < 0).sum())
        if not pd.api.types.is_numeric_dtype(col) and num.notna().any():
            style = "decimal commas" if uses_decimal_comma(col) else "text such as currency symbols or thousands separators"
            add(QualityIssue("info", f"{m}: numbers stored as text",
                             f"Values like {_examples(col.loc[num.notna()], 2)} contain {style}.",
                             "Converted to numbers" + (" (comma read as the decimal point)." if uses_decimal_comma(col) else ".")))
        if n_text:
            add(QualityIssue("warning", f"{m}: {n_text} value(s) are not numbers",
                             "Examples: " + _examples(col.loc[num.isna() & col.notna()]) + ".",
                             "Treated as missing, not guessed."))
        if n_missing:
            share = n_missing / max(len(raw), 1)
            add(QualityIssue("warning" if share >= MISSING_WARN else "info",
                             f"{m}: {n_missing} missing value(s) ({_pct(share)})",
                             "Blank cells." + (" That is a lot, so results for this metric are less reliable." if share >= MISSING_STRONG else ""),
                             "Filled by carrying the previous day's value forward; it is counted and shown after analysis."))
        if n_neg:
            add(QualityIssue("info", f"{m}: {n_neg} negative value(s)",
                             "Below-zero values can be legitimate (refunds, profit, temperature).",
                             "Kept as they are, never replaced with zero."))
    for note in metric_notes(pd.DataFrame({m: numeric[m] for m in metrics}), metrics) if metrics else []:
        add(QualityIssue("warning", note.split(":")[0] + ": " + note.split(": ", 1)[1].split(";")[0],
                         note, "Little or nothing can be learned from it; consider unticking it."))
    if metrics and all(numeric[m].dropna().nunique() <= 1 for m in metrics):
        add(QualityIssue("blocking", "Every selected metric is constant",
                         "A metric that never changes cannot show an anomaly.", "Select columns whose values vary over time."))

    review.mapping_notes = _mapping_notes(raw, profile, date_column, metrics)
    review.columns = _column_table(raw, profile, date_column, metrics, overrides)
    return review


def _mapping_notes(raw: pd.DataFrame, profile: DatasetProfile, date_column: str | None, metrics: list[str]) -> list[str]:
    notes = []
    if len(profile.date_candidates) > 1:
        names = ", ".join(f"'{c}'" for c, _ in profile.date_candidates)
        why = " (its name suggests a date)" if profile.date_column and any(
            t in {"date", "time", "timestamp", "day"} for t in tokens(profile.date_column)) else ""
        notes.append(f"Found {len(profile.date_candidates)} possible time columns: {names}. "
                     f"'{profile.date_column}' was suggested first{why}. Choose another if it is the one that matters.")
    if date_column and profile.date_column and date_column != profile.date_column:
        notes.append(f"You chose '{date_column}' instead of the suggested '{profile.date_column}'.")
    excluded = [c for c in profile.columns if c.is_numeric and not c.usable_metric and c.name != date_column and c.kind != "date"]
    for c in excluded:
        if c.name not in metrics:
            notes.append(f"'{c.name}' holds numbers but was left out: {c.reason}. You can still add it if you disagree.")
    roles_guessed = [m for m in metrics if classify_role(m) != "generic"]
    if roles_guessed:
        notes.append("Metric types (revenue, volume, ...) are guessed from column names. They decide how several rows per day "
                     "are combined and whether a financial metric adds weight to an incident's severity; they never decide "
                     "whether something is an anomaly.")
    return notes


def _column_table(raw: pd.DataFrame, profile: DatasetProfile, date_column: str | None,
                  metrics: list[str], overrides: dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in profile.columns:
        if c.name == date_column:
            used, meaning = "Time column", f"Date/time ({c.date_confidence:.0%} of values read as dates)"
        elif c.name in metrics:
            used = "Metric"
            role = classify_role(c.name)
            agg = overrides.get(c.name) or suggest_aggregation(c.name, role)
            meaning = f"{ROLE_LABELS[role]}; several rows per day are combined by {'adding' if agg == 'sum' else agg}"
        else:
            used = "Not used"
            meaning = profile.ignored.get(c.name) or ("another date column" if c.kind == "date" else c.reason or c.kind)
        rows.append({
            "Column": c.name, "Used as": used, "Why": meaning,
            "Missing": _pct(c.missing_share) if c.missing_share else "0%", "Examples": _examples(raw[c.name]),
        })
    return pd.DataFrame(rows)


def interpreted_preview(raw: pd.DataFrame, date_column: str, metrics: list[str], n: int = 8) -> pd.DataFrame:
    """How the first rows will be read: the date as parsed and each metric as a number."""
    head = raw.head(n)
    out = pd.DataFrame({f"{date_column} (read as)": parse_dates(head[date_column]).dt.strftime("%Y-%m-%d")})
    for m in metrics:
        out[m] = to_numeric(head[m]).to_numpy()
    return out
