"""
Common helpers for TenneT RAW ingestion scripts.

Rules:
- NO silent resampling
- NO feature engineering
- RAW = as close to source as possible
"""

import os
import time
from typing import Callable

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine


BASE_URL = "https://api.tennet.eu"


# ================================
# ENV / CONFIG
# ================================

def get_database_engine():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    return create_engine(database_url)


def get_tennet_api_key() -> str:
    load_dotenv()

    api_key = os.getenv("TENNET_API_KEY")
    if not api_key:
        raise RuntimeError("TENNET_API_KEY is not set")

    return api_key


def build_headers(api_key: str) -> dict:
    return {
        "apiKey": api_key,
        "Accept": "application/json",
    }


# ================================
# DATE HELPERS
# ================================

def fmt_tennet_local(dt: pd.Timestamp) -> str:
    return dt.strftime("%d-%m-%Y %H:%M:%S")


def fmt_tennet_utc_z(dt: pd.Timestamp) -> str:
    return dt.strftime("%d-%m-%Y %H:%M:%SZ")


def month_ranges(start: str, end: str):
    dates = pd.date_range(start=start, end=end, freq="MS")
    return [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]


def day_ranges(start: str, end: str):
    dates = pd.date_range(start=start, end=end, freq="D")
    return [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]


def hour_ranges(start: str, end: str):
    dates = pd.date_range(start=start, end=end, freq="h")
    return [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]


def chunk_4h_ranges_utc(start: str, end: str):
    dates = pd.date_range(start=start, end=end, freq="4h", tz="UTC")
    return [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]


# ================================
# TIME NORMALIZATION
# ================================

def parse_amsterdam_to_utc(series: pd.Series) -> pd.Series:
    """
    Convert Europe/Amsterdam local timestamps to UTC.

    RAW-safe:
    - no resampling
    - no aggregation
    - only timezone conversion

    DST-safe:
    - ambiguous='NaT' avoids pandas crashing on repeated autumn hour
    - nonexistent='shift_forward' handles spring missing hour
    """
    dt = pd.to_datetime(series, errors="coerce")

    localized = dt.dt.tz_localize(
        "Europe/Amsterdam",
        ambiguous="NaT",
        nonexistent="shift_forward",
    )

    return localized.dt.tz_convert("UTC")


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def drop_bad_timestamps(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    bad_timestamps = df[timestamp_col].isna().sum()

    if bad_timestamps:
        print(f"WARNING: dropped {bad_timestamps} invalid/ambiguous DST timestamps")
        df = df.dropna(subset=[timestamp_col])

    return df


# ================================
# VALIDATION
# ================================

def validate_time(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Index must be DatetimeIndex")

    if df.index.tz is None:
        raise RuntimeError("Timestamp must be timezone-aware")

    diffs = df.index.to_series().diff().dropna()

    print("\nTime step distribution:")
    print(diffs.value_counts().head(10))

    duplicates = df.index.duplicated().sum()
    print(f"\nDuplicate timestamps: {duplicates}")

    if not diffs.empty:
        most_common = diffs.mode().iloc[0]
        print(f"Most common step: {most_common}")


# ================================
# DOWNLOAD ENGINE
# ================================

def download_chunks(
    endpoint: str,
    ranges,
    parser_func: Callable[[dict], pd.DataFrame],
    date_formatter: Callable[[pd.Timestamp], str],
    sleep_sec: float = 1.2,
    timeout: int = 120,
    max_retries: int = 3,
) -> pd.DataFrame:
    api_key = get_tennet_api_key()
    headers = build_headers(api_key)
    url = BASE_URL + endpoint

    all_chunks = []
    failed = []

    for i, (start_d, end_d) in enumerate(ranges, 1):
        params = {
            "date_from": date_formatter(start_d),
            "date_to": date_formatter(end_d),
        }

        print(f"[{i}/{len(ranges)}] {params['date_from']} → {params['date_to']}")

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    payload = response.json()
                    df_chunk = parser_func(payload)

                    print("rows:", len(df_chunk))

                    if not df_chunk.empty:
                        all_chunks.append(df_chunk)

                    break

                if response.status_code == 429:
                    wait = sleep_sec * attempt * 10
                    print(f"429 rate limit → sleep {wait}s")
                    time.sleep(wait)
                    continue

                print("FAILED:", response.status_code)
                failed.append(
                    {
                        "params": params,
                        "status_code": response.status_code,
                        "body": response.text[:500],
                    }
                )
                break

            except Exception as exc:
                if attempt == max_retries:
                    print("FAILED:", exc)
                    failed.append(
                        {
                            "params": params,
                            "error": str(exc),
                        }
                    )
                else:
                    wait = sleep_sec * attempt * 5
                    print(f"Exception → sleep {wait}s: {exc}")
                    time.sleep(wait)

        time.sleep(sleep_sec)

    if failed:
        print("\nFailed chunks:", len(failed))
        print(failed[:10])

    if not all_chunks:
        return pd.DataFrame()

    return pd.concat(all_chunks, ignore_index=True).drop_duplicates()


# ================================
# SAVE
# ================================

def save_raw_table(
    df: pd.DataFrame,
    table_name: str,
    source_name: str,
) -> None:
    engine = get_database_engine()

    validate_time(df)

    df_out = df.reset_index()
    df_out["source"] = source_name
    df_out["created_at"] = pd.Timestamp.now(tz="UTC")

    df_out.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
    )

    print(f"\nSaved {len(df_out)} rows to {table_name}")
    print(f"Range: {df_out['timestamp'].min()} → {df_out['timestamp'].max()}")
