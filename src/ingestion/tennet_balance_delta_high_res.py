"""
INGESTION SCRIPT: TenneT balance delta high-res

RAW table:
- raw_tennet_balance_delta_high_res

Rules:
- NO silent resampling
- NO feature engineering
"""

import pandas as pd

from src.ingestion.tennet_common import (
    download_chunks,
    fmt_tennet_local,
    hour_ranges,
    parse_amsterdam_to_utc,
    save_raw_table,
)


TABLE_NAME = "raw_tennet_balance_delta_high_res"
SOURCE_NAME = "tennet_balance_delta_high_res"
ENDPOINT = "/publications/v1/balance-delta-high-res"

START_DATE = "2023-01-01"
END_DATE = "2026-01-01"


def main() -> None:
    print(f"Downloading {TABLE_NAME}...")

    df = load_data()

    if df.empty:
        raise RuntimeError("No data loaded")

    df = normalize(df)

    save_raw_table(df, TABLE_NAME, SOURCE_NAME)


def load_data() -> pd.DataFrame:
    return download_chunks(
        endpoint=ENDPOINT,
        ranges=chunk_4h_ranges_utc(START_DATE, END_DATE),
        parser_func=parse_payload,
        date_formatter=fmt_tennet_utc_z,
        sleep_sec=1.2,
        max_retries=5,
    )


def parse_payload(payload: dict) -> pd.DataFrame:
    """
    TenneT high-res endpoints can differ in shape.
    This parser supports both:
    - Response -> TimeSeries -> Period -> Points
    - list/dict payloads with direct records
    """
    rows = []

    if isinstance(payload, list):
        rows.extend(payload)

    elif isinstance(payload, dict) and "Response" in payload:
        response = payload.get("Response", {})
        timeseries_list = response.get("TimeSeries", [])

        for ts in timeseries_list:
            period = ts.get("Period", {})
            points = period.get("Points", [])

            for point in points:
                rows.append(dict(point))

    elif isinstance(payload, dict):
        # Fallback for flat dict-like responses.
        for key in ["data", "Data", "items", "Items", "records", "Records"]:
            if key in payload and isinstance(payload[key], list):
                rows.extend(payload[key])
                break

    return pd.DataFrame(rows)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ts_col = find_timestamp_column(df)

    df["timestamp"] = parse_utc(df[ts_col])

    drop_cols = [
        "timeInterval_start",
        "timeInterval_end",
        "time_start",
        "time_end",
        "timestamp_utc",
    ]

    for col in df.columns:
        if col not in ["timestamp", *drop_cols]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    df = df.drop(columns=drop_cols, errors="ignore")

    return df.set_index("timestamp").sort_index()


def find_timestamp_column(df: pd.DataFrame) -> str:
    for candidate in [
        "timeInterval_start",
        "time_start",
        "timestamp",
        "timestamp_utc",
        "datetime",
        "dateTime",
    ]:
        if candidate in df.columns:
            return candidate

    raise ValueError(f"No timestamp field found. Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
