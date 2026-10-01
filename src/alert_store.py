"""SQLite alert history: remembers incidents, summaries and what was emailed.

Incidents are keyed by their start date (incidents never overlap, so the
start date is a stable identity across daily runs). Re-running the job on
the same data changes nothing; an incident that grows or changes severity is
updated, its summary is cleared so it gets regenerated, and its notification
state is kept so nobody is emailed twice for the same incident.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LABEL_ORDER = ["Low", "Medium", "High", "Critical"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    key TEXT PRIMARY KEY,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    duration_days INTEGER,
    category TEXT,
    impact TEXT,
    severity_score REAL,
    severity_label TEXT,
    metrics TEXT,
    evidence TEXT,
    pattern TEXT,
    summary TEXT,
    checks TEXT,
    summary_source TEXT,
    first_seen TEXT,
    last_updated TEXT,
    notified_at TEXT
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def incident_key(start_date) -> str:
    return pd.Timestamp(start_date).date().isoformat()


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (and create if needed) the alert database. Use ':memory:' for tests."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def upsert_incidents(conn: sqlite3.Connection, incidents: pd.DataFrame, now: str | None = None) -> list[str]:
    """Insert new incidents, update changed ones, and return keys needing a summary."""
    now = now or _now()
    current_keys = []
    for _, r in incidents.iterrows():
        key = incident_key(r["Start_Date"])
        current_keys.append(key)
        values = {
            "end_date": pd.Timestamp(r["End_Date"]).date().isoformat(),
            "duration_days": int(r["Duration_Days"]),
            "category": r["Category"],
            "impact": r["Impact"],
            "severity_score": float(r["Severity_Score"]),
            "severity_label": r["Severity_Label"],
            "metrics": json.dumps(list(r["Metrics"])),
            "evidence": r["Evidence"],
            "pattern": r["Pattern"],
        }
        existing = conn.execute(
            "SELECT end_date, category, severity_score FROM incidents WHERE key = ?", (key,)
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO incidents (key, start_date, first_seen, last_updated, "
                + ", ".join(values) + ") VALUES (?, ?, ?, ?, " + ", ".join("?" * len(values)) + ")",
                [key, key, now, now, *values.values()],
            )
        elif (existing["end_date"], existing["category"], existing["severity_score"]) != (
            values["end_date"], values["category"], values["severity_score"]
        ):
            assignments = ", ".join(f"{c} = ?" for c in values)
            conn.execute(
                f"UPDATE incidents SET {assignments}, last_updated = ?, "
                "summary = NULL, checks = NULL, summary_source = NULL WHERE key = ?",
                [*values.values(), now, key],
            )
    conn.commit()
    if not current_keys:
        return []
    marks = ",".join("?" * len(current_keys))
    rows = conn.execute(
        f"SELECT key FROM incidents WHERE summary IS NULL AND key IN ({marks}) ORDER BY key", current_keys
    ).fetchall()
    return [row["key"] for row in rows]


def save_summaries(conn: sqlite3.Connection, notes: dict[str, dict]) -> None:
    """Store summaries; `notes` maps incident key -> summarize_incident() result."""
    for key, note in notes.items():
        conn.execute(
            "UPDATE incidents SET summary = ?, checks = ?, summary_source = ? WHERE key = ?",
            (note["summary"], json.dumps(note["checks"]), note["source"], key),
        )
    conn.commit()


def _decode(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else []
    d["checks"] = json.loads(d["checks"]) if d["checks"] else []
    return d


def pending_alerts(conn: sqlite3.Connection, min_label: str = "High", since: str | None = None) -> list[dict]:
    """Un-notified incidents at or above `min_label` (optionally ending on/after `since`)."""
    allowed = LABEL_ORDER[LABEL_ORDER.index(min_label):]
    sql = (
        "SELECT * FROM incidents WHERE notified_at IS NULL AND summary IS NOT NULL "
        f"AND severity_label IN ({','.join('?' * len(allowed))})"
    )
    params: list = list(allowed)
    if since:
        sql += " AND end_date >= ?"
        params.append(since)
    sql += " ORDER BY severity_score DESC"
    return [_decode(r) for r in conn.execute(sql, params).fetchall()]


def mark_notified(conn: sqlite3.Connection, keys: list[str], now: str | None = None) -> None:
    now = now or _now()
    for key in keys:
        conn.execute("UPDATE incidents SET notified_at = ? WHERE key = ?", (now, key))
    conn.commit()


def list_incidents(conn: sqlite3.Connection) -> pd.DataFrame:
    """All stored incidents, newest first."""
    return pd.read_sql_query("SELECT * FROM incidents ORDER BY start_date DESC", conn)
