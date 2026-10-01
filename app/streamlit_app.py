"""KPI Sentinel dashboard.

Run from the project root:
    streamlit run app/streamlit_app.py

Demo mode analyses the bundled dataset; "Upload CSV" analyses your own file
(same columns as data/raw/ecommerce_daily_kpis.csv). Summaries use the
rule-based template unless ANTHROPIC_API_KEY is set, in which case you can
request an AI summary per incident (fact-checked before display).
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.alert_store import connect, list_incidents  # noqa: E402
from src.data_loader import REQUIRED_COLUMNS, SchemaError  # noqa: E402
from src.derived_metrics import ALL_METRICS  # noqa: E402
from src.llm_summary import build_incident_facts, get_default_client, summarize_incident  # noqa: E402
from src.service import analyze  # noqa: E402

DEMO_CSV = Path(os.environ.get("KPI_SENTINEL_DEMO_CSV", ROOT / "data" / "raw" / "ecommerce_daily_kpis.csv"))
ALERT_DB = ROOT / "data" / "processed" / "alerts.db"

st.set_page_config(page_title="KPI Sentinel", page_icon="🛰️", layout="wide")


@st.cache_data(show_spinner="Running detection...")
def run_analysis(raw: pd.DataFrame, window_weeks: int, z_threshold: float):
    return analyze(raw, window_weeks, z_threshold)


@st.cache_data
def load_demo(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def metric_chart(detections: pd.DataFrame, metric: str) -> alt.Chart:
    d = detections.loc[detections["Metric"] == metric, ["Date", "Value", "Baseline_Mean", "Z_Score", "Is_Anomaly"]]
    base = alt.Chart(d).encode(x=alt.X("Date:T", title=None))
    line = base.mark_line().encode(y=alt.Y("Value:Q", title=metric))
    baseline = base.mark_line(strokeDash=[4, 4], color="gray").encode(y="Baseline_Mean:Q")
    flagged = (
        alt.Chart(d[d["Is_Anomaly"]])
        .mark_circle(size=100, color="red")
        .encode(
            x="Date:T",
            y="Value:Q",
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Value:Q", format=",.2f"), alt.Tooltip("Z_Score:Q", format=".1f")],
        )
    )
    return (line + baseline + flagged).properties(height=320)


# ---- sidebar: data and settings --------------------------------------------
st.sidebar.title("KPI Sentinel")
source = st.sidebar.radio("Data source", ["Demo data", "Upload CSV"])
if source == "Demo data":
    raw = load_demo(str(DEMO_CSV))
else:
    uploaded = st.sidebar.file_uploader("Daily KPI CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with columns: " + ", ".join(REQUIRED_COLUMNS))
        st.stop()
    raw = pd.read_csv(uploaded)

window_weeks = st.sidebar.slider("Baseline window (weeks)", min_value=4, max_value=12, value=8)
z_threshold = st.sidebar.slider("z-score threshold", min_value=3.0, max_value=6.0, value=4.0, step=0.5)

try:
    analysis = run_analysis(raw, window_weeks, z_threshold)
except SchemaError as exc:
    st.error(str(exc))
    st.stop()

incidents = analysis.incidents.sort_values("Severity_Score", ascending=False).reset_index(drop=True)

# ---- header -----------------------------------------------------------------
st.title("KPI Sentinel")
st.caption("Statistics decide what is anomalous; the language layer only explains it.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Days analysed", len(analysis.cleaned))
c2.metric("Incidents", len(incidents))
c3.metric("High / Critical", int(incidents["Severity_Label"].isin(["High", "Critical"]).sum()))
c4.metric("Data warnings", len(analysis.validation.warnings))

tab_inc, tab_metric, tab_quality, tab_history, tab_about = st.tabs(
    ["Incidents", "Metric explorer", "Data quality", "Alert history", "About"]
)

# ---- incidents ---------------------------------------------------------------
with tab_inc:
    if incidents.empty:
        st.info("No incidents detected with the current settings.")
    else:
        table = incidents.assign(
            Start=incidents["Start_Date"].dt.date,
            End=incidents["End_Date"].dt.date,
            Severity=incidents["Severity_Label"] + " (" + incidents["Severity_Score"].round(1).astype(str) + ")",
        )[["Start", "End", "Duration_Days", "Category", "Severity", "Evidence"]]
        st.dataframe(table, hide_index=True)

        labels = [f"{r.Start_Date.date()} · {r.Category} · {r.Severity_Label}" for r in incidents.itertuples()]
        choice = st.selectbox("Inspect an incident", range(len(incidents)), format_func=lambda i: labels[i])
        row = incidents.iloc[choice]
        facts = build_incident_facts(row, analysis.detections)

        ai_key = f"ai_{row['Start_Date'].date()}_{window_weeks}_{z_threshold}"
        client = get_default_client()
        if client is not None and st.button("Generate AI summary (Claude)"):
            st.session_state[ai_key] = summarize_incident(facts, client)
        note = st.session_state.get(ai_key) or summarize_incident(facts)

        st.subheader(f"{row['Category']} · {row['Severity_Label']} ({row['Severity_Score']:g})")
        st.write(note["summary"])
        source_note = f"Summary source: {note['source']}"
        if note["problems"]:
            source_note += " (AI output failed fact-checking, so the rule-based summary is shown)"
        st.caption(source_note)
        st.markdown("**Worth checking**")
        for check in note["checks"]:
            st.markdown(f"- {check}")
        with st.expander("Evidence and scoring"):
            st.write(row["Pattern"])
            st.code(row["Evidence"])
            st.caption(row["Severity_Breakdown"])
        if client is None:
            st.caption("Set ANTHROPIC_API_KEY to enable optional AI summaries.")

# ---- metric explorer ---------------------------------------------------------
with tab_metric:
    metric = st.selectbox("Metric", ALL_METRICS)
    st.altair_chart(metric_chart(analysis.detections, metric))
    st.caption("Dashed line: same-weekday baseline. Red dots: days flagged as anomalous.")

# ---- data quality ------------------------------------------------------------
with tab_quality:
    if analysis.validation.is_clean:
        st.success("No data-quality issues found.")
    for warning in analysis.validation.warnings:
        st.warning(warning)
    st.caption("What cleaning changed:")
    st.json(asdict(analysis.cleaning))

# ---- alert history -----------------------------------------------------------
with tab_history:
    if ALERT_DB.exists():
        history = list_incidents(connect(ALERT_DB))
        if history.empty:
            st.info("The alert history is empty.")
        else:
            st.dataframe(
                history[["start_date", "end_date", "category", "severity_label", "severity_score", "summary_source", "notified_at"]],
                hide_index=True,
            )
    else:
        st.info("No alert history yet. Run `python scripts/run_daily.py` to create it.")

# ---- about -------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
**How it works.** Each day and metric is compared with the *same weekday* over
the previous weeks (so ordinary weekend differences are not flagged), using a
z-score plus an IQR check. Flags on consecutive days become one incident,
rule-based logic labels the pattern (e.g. conversion drop, marketing
efficiency), and a heuristic severity score ranks incidents.

**What the language layer does.** It only explains incidents the statistics
already flagged. AI text is fact-checked (numbers must come from the data, no
stated causes) and replaced by a template summary if it fails.

**Limits.** Demo data is synthetic. Short baselines are noisier, the severity
weights are heuristics, and a pattern is not proof of cause.
"""
    )
