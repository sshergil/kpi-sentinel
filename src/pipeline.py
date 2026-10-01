"""End-to-end detection pipeline: cleaned data in, scored incidents out."""

from __future__ import annotations

import pandas as pd

from src.anomaly_detection import Z_THRESHOLD, detect_anomalies
from src.baselines import WINDOW_WEEKS, compute_baselines
from src.business_rules import classify_incidents
from src.derived_metrics import ALL_METRICS, add_derived_metrics
from src.incidents import build_incidents
from src.severity import score_incidents


def run_detection_pipeline(
    cleaned: pd.DataFrame,
    window_weeks: int = WINDOW_WEEKS,
    z_threshold: float = Z_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run derived metrics -> baselines -> detection -> incidents -> rules -> severity.

    `cleaned` must be clean_data() output. Returns (detections, incidents).
    """
    enriched = add_derived_metrics(cleaned)
    baselines = compute_baselines(enriched, metrics=ALL_METRICS, window_weeks=window_weeks)
    detections = detect_anomalies(baselines, z_threshold=z_threshold)
    incidents = score_incidents(classify_incidents(build_incidents(detections)))
    return detections, incidents
