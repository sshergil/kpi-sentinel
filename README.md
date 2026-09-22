# KPI Sentinel

A statistics-first anomaly detection system for e-commerce KPIs that uses an
LLM strictly to translate detected anomalies into business-readable insights
and alerts — never to detect them.

## Status

🚧 Work in progress. This README will be filled out as each milestone lands.

## Milestone 1 — Project Setup + Synthetic Data ✅

- `scripts/generate_synthetic_data.py` generates a reproducible, 365-day
  synthetic daily e-commerce dataset with realistic relationships between
  metrics (Orders = Traffic × Conversion Rate, Revenue = Orders × AOV, etc.),
  weekly seasonality, and a gradual trend.
- 10 known anomalies (mix of single-day and multi-day) are injected
  separately from the clean data generation logic, with a full ground-truth
  log saved alongside the dataset for later evaluation.

### How to run

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
```

This creates:
- `data/raw/ecommerce_daily_kpis.csv`
- `data/raw/ground_truth_anomalies.csv`
