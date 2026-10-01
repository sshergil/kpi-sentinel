"""Email alerts for high-severity incidents (plain text, standard-library SMTP)."""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage


@dataclass
class SmtpConfig:
    host: str
    port: int
    sender: str
    recipients: list[str]
    username: str | None = None
    password: str | None = None
    use_tls: bool = True

    @classmethod
    def from_env(cls) -> "SmtpConfig | None":
        """Read SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_FROM, ALERT_TO.

        Returns None unless host, sender and at least one recipient are set,
        so a missing configuration never crashes the job.
        """
        host, sender = os.environ.get("SMTP_HOST"), os.environ.get("ALERT_FROM")
        recipients = [r.strip() for r in os.environ.get("ALERT_TO", "").split(",") if r.strip()]
        if not (host and sender and recipients):
            return None
        return cls(
            host=host,
            port=int(os.environ.get("SMTP_PORT", "587")),
            sender=sender,
            recipients=recipients,
            username=os.environ.get("SMTP_USER") or None,
            password=os.environ.get("SMTP_PASSWORD") or None,
        )


def build_email(incident: dict, sender: str, recipients: list[str]) -> EmailMessage:
    """Build the alert email for one stored incident (see alert_store)."""
    when = incident["start_date"]
    if incident["end_date"] != incident["start_date"]:
        when += f" to {incident['end_date']}"
    msg = EmailMessage()
    msg["Subject"] = f"[KPI Sentinel] {incident['severity_label']}: {incident['category']} ({when})"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    checks = "\n".join(f"  - {c}" for c in incident["checks"]) or "  - (none)"
    msg.set_content(
        f"Severity: {incident['severity_label']} ({incident['severity_score']:g}/100)\n"
        f"Dates:    {when} ({incident['duration_days']} day(s))\n"
        f"Pattern:  {incident['category']}\n\n"
        f"{incident['summary']}\n\n"
        f"Evidence: {incident['evidence']}\n\n"
        f"Worth checking:\n{checks}\n\n"
        "-- \nFlagged by statistical detection; the explanation above describes patterns "
        "and does not establish a cause."
    )
    return msg


def send_emails(messages: list[EmailMessage], config: SmtpConfig, smtp_factory=smtplib.SMTP) -> int:
    """Send all messages over one SMTP connection. Returns how many were sent."""
    if not messages:
        return 0
    with smtp_factory(config.host, config.port) as server:
        if config.use_tls:
            server.starttls()
        if config.username:
            server.login(config.username, config.password or "")
        for message in messages:
            server.send_message(message)
    return len(messages)
