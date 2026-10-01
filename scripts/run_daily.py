"""Daily batch job: detect, remember, summarize new incidents, email alerts.

Usage (from the project root):
    python scripts/run_daily.py [--data PATH] [--db PATH] [--dry-run]
                                [--min-label High] [--alert-since YYYY-MM-DD]

Without ANTHROPIC_API_KEY, summaries come from the template. Without SMTP_*
settings, due alerts are listed but not sent (and stay pending).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.alert_store import LABEL_ORDER, connect  # noqa: E402
from src.alerting import SmtpConfig  # noqa: E402
from src.daily_job import run_daily  # noqa: E402
from src.llm_summary import get_default_client  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "raw" / "ecommerce_daily_kpis.csv"))
    p.add_argument("--db", default=str(ROOT / "data" / "processed" / "alerts.db"))
    p.add_argument("--dry-run", action="store_true", help="never send email")
    p.add_argument("--min-label", default="High", choices=LABEL_ORDER)
    p.add_argument("--alert-since", default=None, help="only alert for incidents ending on/after this date")
    args = p.parse_args()

    client, smtp = get_default_client(), SmtpConfig.from_env()
    print(f"Summaries: {'Anthropic API' if client else 'template (no API key)'} | "
          f"Email: {'configured' if smtp else 'not configured'}{' (dry run)' if args.dry_run else ''}")

    result = run_daily(
        pd.read_csv(args.data), connect(args.db), client=client, smtp_config=smtp,
        min_alert_label=args.min_label, alert_since=args.alert_since, dry_run=args.dry_run,
    )
    print(f"{result.n_incidents} incidents | {result.n_summarized} newly summarized | "
          f"{result.n_alerts_sent} alerts sent")
    for subject in result.pending_alert_subjects:
        print(f"  pending alert: {subject}")
    if result.pending_alert_subjects:
        print("Alerts above were not sent (dry run or no SMTP configuration) and remain pending.")


if __name__ == "__main__":
    main()
