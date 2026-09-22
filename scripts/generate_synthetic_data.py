"""
scripts/generate_synthetic_data.py

KPI Sentinel — Milestone 1: Synthetic data generation.

This script builds a synthetic daily e-commerce dataset in two clearly
separated stages:

1. generate_clean_data()
   Builds "normal" business data: realistic relationships between metrics,
   weekly seasonality, a gradual trend, and random noise. No anomalies.

2. inject_anomalies()
   Takes the clean data and deliberately breaks it in specific, documented
   ways (single-day and multi-day anomalies). Every injection is logged to
   a ground-truth table so we can later measure whether our anomaly
   detector actually finds what we planted.

Keeping these two stages separate matters: it means the "normal" data
generation logic is never contaminated by anomaly logic, and we always
have an honest, auditable record of exactly what was injected and why.

Run:
    python scripts/generate_synthetic_data.py

Output:
    data/raw/ecommerce_daily_kpis.csv       <- the dataset (with anomalies)
    data/raw/ground_truth_anomalies.csv     <- what was injected, and when
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42          # fixes randomness so results are reproducible
N_DAYS = 365              # ~12 months of daily data
START_DATE = "2025-09-22" # arbitrary anchor date, one year before "today"

# Save relative to the project root, regardless of where this script is run from
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Stage 1: Clean data generation (no anomalies)
# ---------------------------------------------------------------------------

def generate_clean_data(n_days=N_DAYS, start_date=START_DATE, seed=RANDOM_SEED):
    """
    Builds a DataFrame of "normal" daily e-commerce KPIs.

    Design choices (why each metric is built the way it is):
      - Website_Traffic: slow upward trend (business growing) x weekend lift
        (more casual browsing on weekends) + noise.
      - Conversion_Rate: mostly stable, small weekday/weekend pattern + noise.
      - Orders: driven by Traffic x Conversion_Rate, per the project spec.
      - Average_Order_Value: very slow drift + noise.
      - Revenue: driven by Orders x AOV, per the project spec.
      - Ad_Spend: loosely tracks Traffic (spend buys traffic) + noise.
      - Refunds: a percentage of Revenue (a "refund rate") + noise, so
        refunds naturally scale with order/revenue volume.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    day_index = np.arange(n_days)
    weekday = dates.dayofweek  # Monday=0 ... Sunday=6

    # ---- Website Traffic ----
    traffic_trend = 8000 + day_index * 3.0
    weekend_lift = np.where(weekday >= 5, 1.15, 1.0)
    traffic_noise = rng.normal(0, 400, n_days)
    website_traffic = traffic_trend * weekend_lift + traffic_noise
    website_traffic = np.clip(website_traffic, 500, None)

    # ---- Conversion Rate ----
    base_conversion = 0.028
    weekday_conv_lift = np.where(weekday < 5, 1.05, 0.95)
    conversion_noise = rng.normal(0, 0.002, n_days)
    conversion_rate = base_conversion * weekday_conv_lift + conversion_noise
    conversion_rate = np.clip(conversion_rate, 0.005, None)

    # ---- Orders = Traffic x Conversion Rate (+ small independent noise) ----
    orders_noise = rng.normal(0, 3, n_days)
    orders = website_traffic * conversion_rate + orders_noise
    orders = np.clip(orders, 0, None)

    # ---- Average Order Value ----
    aov_trend = 68 + day_index * 0.01
    aov_noise = rng.normal(0, 3, n_days)
    average_order_value = aov_trend + aov_noise
    average_order_value = np.clip(average_order_value, 20, None)

    # ---- Revenue = Orders x AOV (+ small noise) ----
    revenue_noise = rng.normal(0, 150, n_days)
    revenue = orders * average_order_value + revenue_noise
    revenue = np.clip(revenue, 0, None)

    # ---- Ad Spend (loosely tracks traffic) ----
    ad_spend_base = website_traffic * 0.35 + 500
    ad_spend_noise = rng.normal(0, 200, n_days)
    ad_spend = ad_spend_base + ad_spend_noise
    ad_spend = np.clip(ad_spend, 100, None)

    # ---- Refunds (a percentage of Revenue) ----
    base_refund_rate = 0.035
    refund_noise = rng.normal(0, 0.005, n_days)
    refund_rate = np.clip(base_refund_rate + refund_noise, 0.005, None)
    refunds = revenue * refund_rate

    df = pd.DataFrame({
        "Date": dates,
        "Revenue": revenue.round(2),
        "Orders": orders.round(0).astype(int),
        "Website_Traffic": website_traffic.round(0).astype(int),
        "Conversion_Rate": conversion_rate.round(4),
        "Ad_Spend": ad_spend.round(2),
        "Refunds": refunds.round(2),
        "Average_Order_Value": average_order_value.round(2),
    })

    return df


# ---------------------------------------------------------------------------
# Stage 2: Anomaly injection (kept fully separate from clean generation)
# ---------------------------------------------------------------------------

def inject_anomalies(df, seed=RANDOM_SEED):
    """
    Deliberately injects ~10 known anomalies into an otherwise-clean dataset,
    and logs exactly what was done so we have ground truth for evaluation.

    Includes both:
      - single-day anomalies (a spike or drop on one date)
      - multi-day sustained anomalies (a pattern lasting 3-5 days)

    When we change a "driver" metric (e.g. Conversion_Rate), we also
    recompute the metrics that depend on it (e.g. Orders, Revenue), so the
    anomaly propagates the same way a real one would -- this keeps the
    anomalies realistic rather than internally inconsistent.

    Returns:
        df_with_anomalies : DataFrame, same shape as input, with anomalies applied
        ground_truth       : DataFrame documenting every injected anomaly
    """
    rng = np.random.default_rng(seed + 1)  # separate noise stream from clean data
    df = df.copy()
    # Orders/Traffic start as whole-number (int) columns. We need to write
    # floating-point intermediate values into them while injecting anomalies,
    # so we widen them to float64 here and round back to int at the very end.
    df["Orders"] = df["Orders"].astype(float)
    df["Website_Traffic"] = df["Website_Traffic"].astype(float)
    n = len(df)
    ground_truth_records = []

    def log_anomaly(start_idx, end_idx, metrics, anomaly_type, description):
        ground_truth_records.append({
            "start_date": df.loc[start_idx, "Date"].date(),
            "end_date": df.loc[end_idx, "Date"].date(),
            "duration_days": end_idx - start_idx + 1,
            "metrics_affected": ", ".join(metrics),
            "anomaly_type": anomaly_type,
            "description": description,
        })

    # Evenly spaced anchor points, leaving room at the start (for baseline
    # history) and end (so multi-day anomalies don't run off the dataset).
    anchors = np.linspace(40, n - 25, 10).astype(int)

    # 1) Single-day revenue/orders/conversion crash (checkout problem)
    i = anchors[0]
    df.loc[i, "Conversion_Rate"] *= 0.55
    df.loc[i, "Orders"] = df.loc[i, "Website_Traffic"] * df.loc[i, "Conversion_Rate"]
    df.loc[i, "Revenue"] = df.loc[i, "Orders"] * df.loc[i, "Average_Order_Value"]
    log_anomaly(i, i, ["Revenue", "Orders", "Conversion_Rate"], "single_day_drop",
                "Checkout/conversion problem: conversion rate collapses for one day, "
                "dragging orders and revenue down with it.")

    # 2) Single-day traffic spike without proportional orders (low-quality/bot traffic)
    i = anchors[1]
    df.loc[i, "Website_Traffic"] *= 1.9
    log_anomaly(i, i, ["Website_Traffic"], "single_day_spike",
                "Traffic spike (e.g. bot or referral traffic) with no proportional "
                "lift in orders -- traffic quality issue, not a real demand increase.")

    # 3) Multi-day conversion collapse (site bug), 4 days
    start, end = anchors[2], anchors[2] + 3
    for idx in range(start, end + 1):
        df.loc[idx, "Conversion_Rate"] *= 0.6
        df.loc[idx, "Orders"] = df.loc[idx, "Website_Traffic"] * df.loc[idx, "Conversion_Rate"]
        df.loc[idx, "Revenue"] = df.loc[idx, "Orders"] * df.loc[idx, "Average_Order_Value"]
    log_anomaly(start, end, ["Conversion_Rate", "Orders", "Revenue"], "multi_day_sustained",
                "Sustained conversion rate drop over several days, e.g. a broken "
                "checkout step that wasn't fixed immediately.")

    # 4) Single-day refund spike (bad product batch)
    i = anchors[3]
    df.loc[i, "Refunds"] *= 3.2
    log_anomaly(i, i, ["Refunds"], "single_day_spike",
                "Sudden refund spike, e.g. a faulty product batch or shipping issue.")

    # 5) Multi-day ad spend increase without proportional revenue, 5 days
    start, end = anchors[4], anchors[4] + 4
    for idx in range(start, end + 1):
        df.loc[idx, "Ad_Spend"] *= 1.8
        df.loc[idx, "Website_Traffic"] *= 1.15  # traffic ticks up, but not proportionally
        df.loc[idx, "Orders"] = df.loc[idx, "Website_Traffic"] * df.loc[idx, "Conversion_Rate"]
        df.loc[idx, "Revenue"] = df.loc[idx, "Orders"] * df.loc[idx, "Average_Order_Value"]
    log_anomaly(start, end, ["Ad_Spend", "Website_Traffic", "Revenue"], "multi_day_sustained",
                "Ad spend increases sharply for several days while revenue/orders "
                "grow only modestly: declining marketing efficiency.")

    # 6) Multi-day traffic decline (outage / SEO ranking drop), 3 days
    start, end = anchors[5], anchors[5] + 2
    for idx in range(start, end + 1):
        df.loc[idx, "Website_Traffic"] *= 0.6
        df.loc[idx, "Orders"] = df.loc[idx, "Website_Traffic"] * df.loc[idx, "Conversion_Rate"]
        df.loc[idx, "Revenue"] = df.loc[idx, "Orders"] * df.loc[idx, "Average_Order_Value"]
    log_anomaly(start, end, ["Website_Traffic", "Orders", "Revenue"], "multi_day_sustained",
                "Sustained traffic decline (e.g. site outage or search ranking drop) "
                "driving orders and revenue down with it.")

    # 7) Single-day AOV spike (large bulk order / premium promo)
    i = anchors[6]
    df.loc[i, "Average_Order_Value"] *= 1.6
    df.loc[i, "Revenue"] = df.loc[i, "Orders"] * df.loc[i, "Average_Order_Value"]
    log_anomaly(i, i, ["Average_Order_Value", "Revenue"], "single_day_spike",
                "One-day AOV spike, e.g. a bulk or premium purchase, lifting revenue "
                "without a change in order count.")

    # 8) Single-day positive demand spike (flash sale / viral campaign)
    i = anchors[7]
    df.loc[i, "Website_Traffic"] *= 1.5
    df.loc[i, "Conversion_Rate"] *= 1.3
    df.loc[i, "Orders"] = df.loc[i, "Website_Traffic"] * df.loc[i, "Conversion_Rate"]
    df.loc[i, "Revenue"] = df.loc[i, "Orders"] * df.loc[i, "Average_Order_Value"]
    log_anomaly(i, i, ["Website_Traffic", "Conversion_Rate", "Orders", "Revenue"],
                "single_day_spike",
                "Positive demand spike, e.g. a flash sale or viral moment: traffic, "
                "conversion, orders and revenue all jump together. A good anomaly, "
                "but still statistically unusual.")

    # 9) Multi-day refund rate increase (recurring product quality issue), 4 days
    start, end = anchors[8], anchors[8] + 3
    for idx in range(start, end + 1):
        df.loc[idx, "Refunds"] *= 2.1
    log_anomaly(start, end, ["Refunds"], "multi_day_sustained",
                "Sustained elevated refund rate over several days, e.g. a recurring "
                "product defect surfacing across many orders.")

    # 10) Multi-day broad decline, traffic stable (Scenario A), 3 days
    start, end = anchors[9], anchors[9] + 2
    for idx in range(start, end + 1):
        df.loc[idx, "Conversion_Rate"] *= 0.7
        df.loc[idx, "Orders"] = df.loc[idx, "Website_Traffic"] * df.loc[idx, "Conversion_Rate"]
        df.loc[idx, "Revenue"] = df.loc[idx, "Orders"] * df.loc[idx, "Average_Order_Value"]
    log_anomaly(start, end, ["Revenue", "Orders", "Conversion_Rate"], "multi_day_sustained",
                "Broad sustained decline in revenue, orders, and conversion while "
                "traffic stays roughly stable -- points at a purchasing-behavior "
                "problem rather than a traffic problem.")

    # Re-round everything after modifications so the final file stays tidy
    df["Orders"] = df["Orders"].round(0).astype(int)
    df["Website_Traffic"] = df["Website_Traffic"].round(0).astype(int)
    df["Revenue"] = df["Revenue"].round(2)
    df["Conversion_Rate"] = df["Conversion_Rate"].round(4)
    df["Ad_Spend"] = df["Ad_Spend"].round(2)
    df["Refunds"] = df["Refunds"].round(2)
    df["Average_Order_Value"] = df["Average_Order_Value"].round(2)

    ground_truth = pd.DataFrame(ground_truth_records)
    return df, ground_truth


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = generate_clean_data()
    final_df, ground_truth_df = inject_anomalies(clean_df)

    data_path = OUTPUT_DIR / "ecommerce_daily_kpis.csv"
    truth_path = OUTPUT_DIR / "ground_truth_anomalies.csv"

    final_df.to_csv(data_path, index=False)
    ground_truth_df.to_csv(truth_path, index=False)

    print(f"Saved {len(final_df)} days of data to: {data_path}")
    print(f"Saved {len(ground_truth_df)} ground-truth anomaly records to: {truth_path}")
    print("\nPreview of dataset (first 5 rows):")
    print(final_df.head().to_string(index=False))
    print("\nInjected anomalies:")
    print(ground_truth_df.to_string(index=False))


if __name__ == "__main__":
    main()
