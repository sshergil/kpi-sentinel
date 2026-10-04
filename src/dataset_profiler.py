"""Inspect an arbitrary CSV and infer its time column and usable metrics.

Everything here is plain pandas/statistical heuristics: no LLM. The profile
is advice shown to the user, who can override any choice before analysis.

Semantic roles (revenue, spend, volume, ...) are inferred from column names
for context and presentation only; they never influence whether something is
flagged as anomalous.
"""

from __future__ import annotations

import csv
import io
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATE_NAME_HINTS = {"date", "time", "timestamp", "datetime", "day", "month", "week", "period", "dt"}
DATE_PLAUSIBLE = 0.6  # minimum confidence to offer a column as a time column
NUMERIC_SHARE_MIN = 0.9  # share of non-null values that must parse as numbers
MOSTLY_MISSING = 0.5
NEAR_CONSTANT = 0.98
SAMPLE_SIZE = 2000
PLAUSIBLE_MIN = pd.Timestamp("1900-01-01")
PLAUSIBLE_MAX = pd.Timestamp("2100-12-31")

ID_TOKENS = {"id", "ids", "uuid", "guid", "key", "sku", "index", "idx", "zip", "zipcode", "postal", "phone", "pk"}
ID_SUFFIX = re.compile(
    r"(customer|order|user|product|transaction|txn|item|account|client|session|store|invoice|employee|"
    r"member|visitor|ticket|booking|device|row|record|seller|vendor|campaign)id$"
)
CALENDAR_TOKENS = {"year", "month", "week", "weekday", "dow", "quarter", "hour", "minute", "dayofweek", "dayofmonth"}

# Ordered: the first role whose keywords match a column-name token wins.
ROLE_KEYWORDS: list[tuple[str, set[str]]] = [
    ("refund_return", {"refund", "refunds", "return", "returns", "returned", "chargeback", "chargebacks", "cancel", "cancellation", "cancellations"}),
    ("conversion", {"conversion", "conversions", "conv", "cvr"}),
    ("spend", {"spend", "marketing", "advertising", "ad", "ads", "adspend", "campaign"}),
    ("cost", {"cost", "costs", "expense", "expenses", "cogs", "overhead"}),
    ("percentage", {"pct", "percent", "percentage", "share"}),
    ("rate", {"rate", "ratio", "ctr", "bounce", "churn"}),
    ("revenue", {"revenue", "sales", "income", "turnover", "gmv", "earnings", "amount", "profit", "aov", "value", "price", "basket"}),
    ("traffic", {"traffic", "visits", "sessions", "pageviews", "impressions", "clicks", "hits"}),
    ("volume", {"customers", "customer", "users", "user", "visitors", "visitor", "orders", "order", "transactions", "transaction",
                "units", "unit", "quantity", "qty", "count", "sold", "items", "bookings", "tickets"}),
    ("inventory", {"inventory", "stock", "backlog", "onhand"}),
    ("engagement", {"engagement", "likes", "comments", "followers", "subscribers", "reactions", "logins", "dau", "mau"}),
]
ROLE_LABELS = {
    "revenue": "Financial (revenue)", "spend": "Spend", "cost": "Cost", "volume": "Volume",
    "traffic": "Traffic", "conversion": "Conversion", "rate": "Rate", "percentage": "Percentage",
    "refund_return": "Refund / return", "inventory": "Inventory", "engagement": "Engagement",
    "generic": "Generic numeric metric",
}
# Suggested aggregation when several rows share one date (levels average, amounts add up).
SUM_ROLES = {"revenue", "spend", "cost", "volume", "traffic", "refund_return", "engagement"}


@dataclass
class ColumnProfile:
    name: str
    kind: str  # date | numeric | categorical | text | identifier | binary | calendar | empty | constant | mostly_missing
    missing_share: float
    n_unique: int
    date_confidence: float = 0.0
    role: str | None = None  # semantic role, numeric columns only
    suggested_agg: str | None = None
    is_numeric: bool = False  # values parse as numbers (a candidate the user may still select)
    usable_metric: bool = False
    reason: str = ""  # why a column is ignored, or a caution


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    date_candidates: list[tuple[str, float]] = field(default_factory=list)  # best first
    date_column: str | None = None
    date_ambiguous: bool = False
    metric_columns: list[str] = field(default_factory=list)  # default selection
    ignored: dict[str, str] = field(default_factory=dict)

    def column(self, name: str) -> ColumnProfile:
        return next(c for c in self.columns if c.name == name)

    @property
    def problem(self) -> str | None:
        """Why this dataset cannot be analyzed as-is, or None."""
        if self.date_column is None and not self.metric_columns:
            return ("KPI Sentinel couldn't find a usable time column and numeric metrics in this dataset. "
                    "Please upload a dataset containing a date/time column and at least one numeric measure.")
        if self.date_column is None:
            return ("Numeric metrics were found, but a time column is required for time-series anomaly detection.")
        if not self.metric_columns:
            return "A time column was detected, but no numeric KPI columns were found."
        return None


@dataclass
class TimeAxis:
    """How a chosen date column is laid out in time."""

    n_valid: int
    n_invalid: int
    n_unique_dates: int
    rows_per_date: float  # median
    median_gap_days: float | None
    frequency: str  # daily | weekly | monthly | irregular | too_short
    missing_day_share: float  # share of calendar days between first and last date with no row
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None

    @property
    def multiple_rows_per_date(self) -> bool:
        return self.rows_per_date >= 2

    @property
    def supported(self) -> bool:
        return self.frequency == "daily"

    @property
    def problem(self) -> str | None:
        if self.frequency == "daily":
            return None
        if self.frequency == "too_short":
            return "The selected time column has fewer than 2 distinct dates, so there is no time series to analyze."
        names = {"weekly": "weekly", "monthly": "monthly", "irregular": "irregularly spaced"}
        return (f"This data looks {names[self.frequency]} (typical gap {self.median_gap_days:g} days). "
                "KPI Sentinel's detector currently needs daily data (one row per day, or several rows per day "
                "that can be aggregated). Weekly and monthly data are not supported yet.")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes, sniffing the delimiter and trying common encodings."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            sep = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
        return pd.read_csv(io.StringIO(text), sep=sep)
    raise ValueError(f"Could not decode the file as text: {last_error}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def tokens(name: str) -> list[str]:
    """Lower-case word tokens of a column name (splits on punctuation and camelCase)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def to_numeric(series: pd.Series) -> pd.Series:
    """Parse numbers stored as text ('1,200', '$3.50', '45%', ' 12 '); unparseable -> NaN."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (series.astype("string").str.strip()
               .str.replace(r"[\$€£,%\s]", "", regex=True)
               .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a column to datetimes (NaT where impossible), normalized to midnight."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series).dt.normalize()
    if pd.api.types.is_numeric_dtype(series) or _looks_numeric(series):
        # Only YYYYMMDD integers count as dates; other numbers are not dates.
        text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        ok = text.str.fullmatch(r"(19|20)\d{6}").fillna(False)
        parsed = pd.to_datetime(text.where(ok), format="%Y%m%d", errors="coerce")
        return parsed.dt.normalize()
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kwargs in ({}, {"format": "mixed"}, {"utc": True}, {"format": "mixed", "utc": True}):
            try:
                parsed = pd.to_datetime(series, errors="coerce", **kwargs)
            except Exception:  # a variant that cannot handle this column is simply skipped
                continue
            if best is None or parsed.notna().sum() > best.notna().sum():
                best = parsed
    if best is None:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if getattr(best.dt, "tz", None) is not None:
        best = best.dt.tz_localize(None)
    # Implausible years (e.g. 'T00001' parsing as year 1) are not real dates.
    best = best.where((best >= PLAUSIBLE_MIN) & (best <= PLAUSIBLE_MAX))
    return best.dt.normalize()


def metric_notes(frame: pd.DataFrame, metrics: list[str]) -> list[str]:
    """Quality cautions for the chosen metrics (constant, near-constant, sparse)."""
    notes = []
    for m in metrics:
        s = frame[m].dropna()
        if s.empty:
            notes.append(f"{m}: no values; it cannot be analyzed")
        elif s.nunique() <= 1:
            notes.append(f"{m}: constant metric (every value is the same); no anomalies can be detected")
        elif s.value_counts(normalize=True).iloc[0] >= NEAR_CONSTANT:
            notes.append(f"{m}: near-constant ({s.value_counts(normalize=True).iloc[0]:.0%} identical values); detection is unreliable")
        missing = float(frame[m].isna().mean())
        if missing >= 0.2:
            notes.append(f"{m}: {missing:.0%} of values are missing")
    return notes


def _looks_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    return to_numeric(non_null).notna().mean() >= NUMERIC_SHARE_MIN


def _name_hint(name: str) -> bool:
    return any(t in DATE_NAME_HINTS for t in tokens(name))


def date_confidence(series: pd.Series, name: str) -> float:
    """0..1 confidence that a column is a date/time column."""
    sample = series.dropna()
    if sample.empty:
        return 0.0
    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(SAMPLE_SIZE, random_state=0)
    parsed = parse_dates(sample)
    rate = float(parsed.notna().mean())
    if parsed.dropna().nunique() < 2:
        return 0.0
    base = rate if rate >= 0.8 else rate * 0.5
    return round(min(1.0, base * 0.85 + (0.15 if _name_hint(name) else 0.0)), 3)


def classify_role(name: str) -> str:
    toks = set(tokens(name))
    squashed = "".join(tokens(name))
    for role, words in ROLE_KEYWORDS:
        if toks & words or squashed in words:
            return role
    return "generic"


def _is_identifier_name(name: str) -> bool:
    toks = tokens(name)
    return bool(set(toks) & ID_TOKENS) or bool(ID_SUFFIX.search("".join(toks)))


def _is_calendar_name(name: str) -> bool:
    toks = tokens(name)
    return bool(toks) and (set(toks) <= CALENDAR_TOKENS or "".join(toks) in CALENDAR_TOKENS or toks == ["day", "of", "week"])


def _is_sequential(values: pd.Series) -> bool:
    v = values.dropna()
    if len(v) < 10 or not (v == v.round()).all():
        return False
    steps = np.diff(v.to_numpy())
    return bool((steps == steps[0]).all() and abs(steps[0]) == 1)


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------

def _profile_column(df: pd.DataFrame, name: str) -> ColumnProfile:
    s = df[name]
    missing = float(s.isna().mean()) if len(s) else 1.0
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    p = ColumnProfile(name=name, kind="text", missing_share=round(missing, 3), n_unique=n_unique)

    if non_null.empty:
        p.kind, p.reason = "empty", "column is empty"
        return p

    conf = 0.0 if pd.api.types.is_numeric_dtype(s) and not _name_hint(name) else date_confidence(s, name)
    if pd.api.types.is_numeric_dtype(s) or _looks_numeric(s):
        yyyymmdd = conf >= DATE_PLAUSIBLE
        if yyyymmdd:
            p.kind, p.date_confidence = "date", conf
            return p
        return _profile_numeric(p, s, non_null)

    if conf >= DATE_PLAUSIBLE:
        p.kind, p.date_confidence = "date", conf
        return p

    unique_ratio = n_unique / len(non_null)
    avg_len = float(non_null.astype(str).str.len().mean())
    if _is_identifier_name(name) or (unique_ratio > 0.9 and n_unique > 50 and avg_len < 40):
        p.kind, p.reason = "identifier", "looks like an identifier or label"
    elif avg_len > 40:
        p.kind, p.reason = "text", "free text"
    else:
        p.kind, p.reason = "categorical", "categories, not a numeric measure"
    return p


def _profile_numeric(p: ColumnProfile, s: pd.Series, non_null: pd.Series) -> ColumnProfile:
    values = to_numeric(non_null).dropna()
    top_share = float(values.value_counts(normalize=True).iloc[0]) if len(values) else 1.0
    p.n_unique = int(values.nunique())
    p.kind = "numeric"
    p.is_numeric = True
    p.role = classify_role(p.name)
    p.suggested_agg = "sum" if p.role in SUM_ROLES else "mean"

    if p.missing_share >= MOSTLY_MISSING:
        p.kind, p.reason = "mostly_missing", f"{p.missing_share:.0%} of values are missing"
    elif p.n_unique <= 1:
        p.kind, p.reason = "constant", "every value is the same"
    elif _is_identifier_name(p.name) or _is_sequential(values):
        p.kind, p.reason = "identifier", "looks like an identifier or row counter"
    elif _is_calendar_name(p.name):
        p.kind, p.reason = "calendar", "calendar component, not a KPI"
    elif p.n_unique == 2:
        p.kind, p.reason = "binary", "binary flag, not a measure"
    elif top_share >= NEAR_CONSTANT:
        p.kind, p.reason = "constant", f"{top_share:.0%} of values are identical"
    else:
        p.usable_metric = True
        if p.missing_share > 0.2:
            p.reason = f"{p.missing_share:.0%} of values are missing"
    if p.kind != "numeric":
        p.role = p.role if p.kind in {"constant", "mostly_missing"} else None
    return p


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Profile every column and propose a time column and metric columns."""
    columns = [_profile_column(df, str(c)) for c in df.columns]
    profile = DatasetProfile(n_rows=len(df), n_columns=len(df.columns), columns=columns)

    candidates = [(c.name, c.date_confidence) for c in columns if c.kind == "date"]
    position = {c.name: i for i, c in enumerate(columns)}
    candidates.sort(key=lambda t: (-t[1], not _name_hint(t[0]), position[t[0]]))
    profile.date_candidates = candidates
    if candidates:
        profile.date_column = candidates[0][0]
        profile.date_ambiguous = len(candidates) > 1

    profile.metric_columns = [c.name for c in columns if c.usable_metric]
    profile.ignored = {c.name: c.reason for c in columns
                       if not c.usable_metric and c.name != profile.date_column and c.reason}
    for c in columns:  # other date-like columns are not metrics
        if c.kind == "date" and c.name != profile.date_column:
            profile.ignored[c.name] = "another date column"
    return profile


def describe_time_axis(df: pd.DataFrame, date_column: str) -> TimeAxis:
    """Frequency, duplicates and gaps for the chosen date column."""
    parsed = parse_dates(df[date_column])
    valid = parsed.dropna()
    n_invalid = int(parsed.isna().sum())
    unique = valid.drop_duplicates().sort_values()
    if len(unique) < 2:
        return TimeAxis(len(valid), n_invalid, len(unique), float(len(valid)), None, "too_short", 0.0)

    gaps = unique.diff().dropna().dt.days
    median_gap = float(gaps.median())
    if median_gap <= 1:
        freq = "daily"
    elif 6 <= median_gap <= 8:
        freq = "weekly"
    elif 27 <= median_gap <= 32:
        freq = "monthly"
    else:
        freq = "irregular"

    span = (unique.iloc[-1] - unique.iloc[0]).days + 1
    missing_share = 1 - len(unique) / span
    if freq == "daily" and missing_share > 0.5:
        freq = "irregular"  # too many days missing to treat as a daily series

    return TimeAxis(
        n_valid=len(valid), n_invalid=n_invalid, n_unique_dates=len(unique),
        rows_per_date=float(valid.value_counts().median()), median_gap_days=median_gap,
        frequency=freq, missing_day_share=round(missing_share, 3),
        first_date=unique.iloc[0], last_date=unique.iloc[-1],
    )
