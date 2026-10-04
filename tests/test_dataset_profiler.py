import numpy as np
import pandas as pd

from src.dataset_profiler import (
    classify_role, date_confidence, describe_time_axis, parse_dates, profile_dataset,
    read_csv_bytes, to_numeric,
)
from tests.helpers import ECOM_LIKE, WEATHER, upload_frame


def kinds(profile):
    return {c.name: c.kind for c in profile.columns}


# ---- Dataset A: e-commerce-like with arbitrary names -----------------------
def test_dataset_a_ecommerce_with_custom_names():
    p = profile_dataset(upload_frame({"Sales": 12000, "Customers": 450, "Ad_Spend": 1800}))
    assert p.date_column == "Date" and p.problem is None
    assert p.metric_columns == ["Sales", "Customers", "Ad_Spend"]
    assert p.column("Date").date_confidence >= 0.9
    assert p.column("Sales").date_confidence == 0.0
    assert p.column("Sales").role == "revenue"
    assert p.column("Customers").role == "volume"
    assert p.column("Ad_Spend").role == "spend"


# ---- Dataset B: a completely different domain --------------------------------
def test_dataset_b_weather_with_timestamp_column():
    df = upload_frame(WEATHER, date_name="timestamp", seasonal=False)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Temperature"] -= 25  # negative temperatures are legitimate
    p = profile_dataset(df)
    assert p.date_column == "timestamp"
    assert p.metric_columns == ["Temperature", "Humidity", "Pressure"]
    assert {p.column(c).role for c in p.metric_columns} == {"generic"}


# ---- Dataset C: identifiers --------------------------------------------------
def test_dataset_c_identifier_columns_are_ignored():
    df = upload_frame({"Sales": 12000})
    rng = np.random.default_rng(1)
    df["Customer_ID"] = rng.permutation(np.arange(1001, 1001 + len(df)))
    df["transactionId"] = [f"T{i:05d}" for i in range(len(df))]
    p = profile_dataset(df)
    assert p.metric_columns == ["Sales"]
    assert "identifier" in p.ignored["Customer_ID"]
    assert "identifier" in p.ignored["transactionId"]


def test_row_counter_column_is_an_identifier():
    df = upload_frame({"Sales": 12000})
    df.insert(0, "Unnamed: 0", np.arange(len(df)))
    assert "Unnamed: 0" not in profile_dataset(df).metric_columns


# ---- Dataset D: no date --------------------------------------------------------
def test_dataset_d_no_time_column():
    df = upload_frame({"Sales": 12000, "Customers": 450, "Revenue": 9000}).drop(columns=["Date"])
    p = profile_dataset(df)
    assert p.date_column is None and p.metric_columns
    assert "time column is required" in p.problem


# ---- Dataset E: no numeric columns ------------------------------------------------
def test_dataset_e_no_numeric_metrics():
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=50),
                       "City": ["Baltimore", "DC"] * 25, "Category": ["a", "b", "c", "d", "e"] * 10})
    p = profile_dataset(df)
    assert p.date_column == "Date" and p.metric_columns == []
    assert p.problem == "A time column was detected, but no numeric KPI columns were found."


def test_unsuitable_dataset_gets_a_friendly_message_not_a_crash():
    df = pd.DataFrame({"Name": ["John", "Sarah", "Ana"], "City": ["Baltimore", "DC", "NYC"],
                       "Favorite_Color": ["Blue", "Green", "Red"]})
    p = profile_dataset(df)
    assert p.date_column is None and p.metric_columns == []
    assert "couldn't find a usable time column and numeric metrics" in p.problem


# ---- Dataset F: messy ---------------------------------------------------------------
def test_dataset_f_messy_values_are_handled():
    n = 100
    dates = pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    dates[10] = "not a date"
    dates[20] = None
    sales = [f"${12000 + i * 7:,.2f}" for i in range(n)]  # numbers stored as strings
    sales[5] = None
    returns = [str(30 + (i * 7) % 11) for i in range(n)]
    df = pd.DataFrame({"Date": dates, "Sales": sales, "Returns": returns, "Share": [f"{40 + i % 9}%" for i in range(n)]})
    df = pd.concat([df, df.iloc[[3, 3]]], ignore_index=True)  # duplicate rows
    p = profile_dataset(df)
    assert p.date_column == "Date"
    assert {"Sales", "Returns", "Share"} <= set(p.metric_columns)
    assert p.column("Sales").role == "revenue" and p.column("Share").role == "percentage"
    assert p.column("Sales").missing_share > 0


def test_string_numbers_are_parsed():
    s = pd.Series(["1,200", "$3.50", "45%", "(12)", " 7 ", "abc", None])
    out = to_numeric(s)
    assert list(out.iloc[:5]) == [1200.0, 3.5, 45.0, -12.0, 7.0]
    assert out.iloc[5:].isna().all()


# ---- dates ---------------------------------------------------------------------------
def test_numbers_are_never_mistaken_for_dates():
    df = upload_frame({"Sales": 12000, "Customers": 450})
    df["Count_As_Text"] = (df["Customers"].round().astype(int)).astype(str)
    p = profile_dataset(df)
    assert p.date_column == "Date" and not p.date_ambiguous
    assert date_confidence(df["Sales"], "Sales") == 0.0
    assert date_confidence(df["Count_As_Text"], "Count_As_Text") == 0.0


def test_multiple_date_columns_are_ambiguous_and_ranked():
    df = upload_frame({"Sales": 12000}, date_name="Order_Date")
    df["Ship_Date"] = df["Order_Date"] + pd.Timedelta(days=2)
    p = profile_dataset(df)
    assert p.date_ambiguous and [c for c, _ in p.date_candidates] == ["Order_Date", "Ship_Date"]
    assert p.date_column == "Order_Date" and "Ship_Date" not in p.metric_columns
    assert p.ignored["Ship_Date"] == "another date column"


def test_name_hint_breaks_ties():
    df = upload_frame({"Sales": 12000})
    df["Created"] = df["Date"].dt.strftime("%d/%m/%Y")
    df = df.rename(columns={"Date": "Day"})
    assert profile_dataset(df).date_column == "Day"


def test_yyyymmdd_integers_count_as_dates():
    df = upload_frame({"Sales": 12000})
    df["Date"] = df["Date"].dt.strftime("%Y%m%d").astype(int)
    assert profile_dataset(df).date_column == "Date"


# ---- ignored numeric columns ------------------------------------------------------------
def test_calendar_binary_constant_and_mostly_missing_columns_are_ignored():
    df = upload_frame({"Sales": 12000})
    df["Month"] = df["Date"].dt.month
    df["Is_Promo"] = (np.arange(len(df)) % 5 == 0).astype(int)
    df["Currency_Rate"] = 1.0
    df["Sparse"] = np.where(np.arange(len(df)) % 10 == 0, 5.0, np.nan)
    p = profile_dataset(df)
    assert p.metric_columns == ["Sales"]
    assert "calendar" in p.ignored["Month"] and "binary" in p.ignored["Is_Promo"]
    assert "same" in p.ignored["Currency_Rate"] and "missing" in p.ignored["Sparse"]


def test_small_count_metric_is_still_offered():
    df = upload_frame({"Sales": 12000})
    df["Returns"] = np.random.default_rng(0).integers(0, 9, len(df))  # few distinct values, but a real KPI
    assert "Returns" in profile_dataset(df).metric_columns


# ---- roles ---------------------------------------------------------------------------------
def test_semantic_roles():
    expected = {
        "Total_Sales": "revenue", "Net_Income": "revenue", "Average_Order_Value": "revenue",
        "Customers": "volume", "Units_Sold": "volume", "CustomerCount": "volume",
        "Marketing_Cost": "spend", "adSpend": "spend", "Ad_Spend": "spend", "Cost": "cost",
        "Sessions": "traffic", "Website_Traffic": "traffic", "Conversion_Rate": "conversion",
        "Bounce_Rate": "rate", "Discount_Pct": "percentage", "Returns": "refund_return",
        "Refunds": "refund_return", "Stock_Level": "inventory", "Followers": "engagement",
        "Temperature": "generic", "Humidity": "generic",
    }
    for name, role in expected.items():
        assert classify_role(name) == role, name


# ---- time axis -----------------------------------------------------------------------------------
def test_daily_axis():
    a = describe_time_axis(upload_frame({"Sales": 1}), "Date")
    assert a.frequency == "daily" and a.supported and a.problem is None and not a.multiple_rows_per_date


def test_hourly_rows_are_daily_with_multiple_rows_per_date():
    hourly = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=24 * 40, freq="h"), "Sales": 1.0})
    a = describe_time_axis(hourly, "ts")
    assert a.frequency == "daily" and a.multiple_rows_per_date and a.rows_per_date == 24


def test_weekly_monthly_and_irregular_are_not_supported_yet():
    weekly = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=60, freq="7D")})
    monthly = pd.DataFrame({"Date": pd.date_range("2018-01-01", periods=60, freq="MS")})
    irregular = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01", "2025-03-20", "2025-07-01", "2025-12-30"])})
    for df, word in ((weekly, "weekly"), (monthly, "monthly"), (irregular, "irregular")):
        a = describe_time_axis(df, "Date")
        assert not a.supported and word in a.problem and "daily data" in a.problem


def test_business_day_data_is_accepted_with_missing_days_reported():
    bdays = pd.DataFrame({"Date": pd.bdate_range("2025-01-01", periods=120)})
    a = describe_time_axis(bdays, "Date")
    assert a.frequency == "daily" and 0.25 < a.missing_day_share < 0.35


def test_long_gaps_make_it_irregular_even_if_most_steps_are_one_day():
    dates = list(pd.date_range("2025-01-01", periods=60)) + list(pd.date_range("2025-09-01", periods=5))
    a = describe_time_axis(pd.DataFrame({"Date": dates}), "Date")
    assert a.median_gap_days == 1 and a.missing_day_share > 0.5
    assert a.frequency == "irregular" and not a.supported


def test_single_date_is_too_short():
    a = describe_time_axis(pd.DataFrame({"Date": ["2025-01-01"] * 5}), "Date")
    assert a.frequency == "too_short" and "fewer than 2 distinct dates" in a.problem


def test_invalid_dates_are_counted():
    df = pd.DataFrame({"Date": ["2025-01-01", "2025-01-02", "oops", "2025-01-04"]})
    assert describe_time_axis(df, "Date").n_invalid == 1


def test_parse_dates_handles_mixed_formats_and_timezones():
    s = pd.Series(["2025-01-01", "2025-01-02 10:30:00", "2025-01-03T08:00:00Z"])
    out = parse_dates(s)
    assert out.notna().all() and out.dt.tz is None


# ---- reading --------------------------------------------------------------------------------------
def test_read_csv_bytes_sniffs_delimiters_and_encodings():
    semicolon = "Date;Sales\n2025-01-01;10\n2025-01-02;12\n".encode("utf-8")
    tabbed = "Date\tSales\n2025-01-01\t10\n".encode("utf-8")
    latin = "Date,Café\n2025-01-01,10\n".encode("latin-1")
    assert list(read_csv_bytes(semicolon).columns) == ["Date", "Sales"]
    assert list(read_csv_bytes(tabbed).columns) == ["Date", "Sales"]
    assert list(read_csv_bytes(latin).columns) == ["Date", "Café"]
