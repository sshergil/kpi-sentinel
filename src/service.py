"""One entry point that turns raw KPI data into detections and incidents.

Shared by the batch job (scripts/run_daily.py) and the Streamlit dashboard,
so both always apply exactly the same validate -> clean -> detect pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.anomaly_detection import Z_THRESHOLD
from src.baselines import WINDOW_WEEKS
from src.data_cleaning import CleaningReport, clean_data
from src.data_loader import ValidationReport, validate_schema
from src.pipeline import run_detection_pipeline


@dataclass
class Analysis:
    cleaned: pd.DataFrame
    detections: pd.DataFrame
    incidents: pd.DataFrame
    validation: ValidationReport
    cleaning: CleaningReport


def analyze(
    raw: pd.DataFrame, window_weeks: int = WINDOW_WEEKS, z_threshold: float = Z_THRESHOLD
) -> Analysis:
    """Validate, clean and analyse raw data. Raises SchemaError if columns are missing."""
    validation = validate_schema(raw)
    cleaned, cleaning = clean_data(raw)
    detections, incidents = run_detection_pipeline(cleaned, window_weeks, z_threshold)
    return Analysis(cleaned, detections, incidents, validation, cleaning)
