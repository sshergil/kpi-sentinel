"""The daily batch job: analyse data, remember incidents, summarize new ones, email alerts.

Idempotent by design: running it twice on the same data summarizes nothing
new and sends no duplicate emails.
"""

from __future__ import annotations

import smtplib
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from src.alert_store import mark_notified, pending_alerts, save_summaries, upsert_incidents, incident_key
from src.alerting import SmtpConfig, build_email, send_emails
from src.llm_summary import LLMClient, build_incident_facts, summarize_incident
from src.service import analyze


@dataclass
class DailyResult:
    n_incidents: int = 0
    n_summarized: int = 0
    n_alerts_sent: int = 0
    pending_alert_subjects: list[str] = field(default_factory=list)  # due but not sent


def run_daily(
    raw: pd.DataFrame,
    conn: sqlite3.Connection,
    client: LLMClient | None = None,
    smtp_config: SmtpConfig | None = None,
    min_alert_label: str = "High",
    alert_since: str | None = None,
    dry_run: bool = False,
    smtp_factory=smtplib.SMTP,
    now: str | None = None,
) -> DailyResult:
    """Run the job. Alerts are only marked as sent after they were actually sent."""
    analysis = analyze(raw)
    result = DailyResult(n_incidents=len(analysis.incidents))

    needs_summary = set(upsert_incidents(conn, analysis.incidents, now))
    notes = {}
    for _, row in analysis.incidents.iterrows():
        key = incident_key(row["Start_Date"])
        if key in needs_summary:
            notes[key] = summarize_incident(build_incident_facts(row, analysis.detections), client)
    save_summaries(conn, notes)
    result.n_summarized = len(notes)

    pending = pending_alerts(conn, min_alert_label, alert_since)
    sender = smtp_config.sender if smtp_config else "kpi-sentinel@localhost"
    recipients = smtp_config.recipients if smtp_config else []
    messages = [build_email(p, sender, recipients) for p in pending]
    result.pending_alert_subjects = [m["Subject"] for m in messages]

    if smtp_config and not dry_run and messages:
        result.n_alerts_sent = send_emails(messages, smtp_config, smtp_factory)
        mark_notified(conn, [p["key"] for p in pending], now)
        result.pending_alert_subjects = []
    return result
