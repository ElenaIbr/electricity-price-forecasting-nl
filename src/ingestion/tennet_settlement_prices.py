"""
INGESTION SCRIPT: TenneT settlement prices

RAW table:
- raw_tennet_settlement_prices

Rules:
- NO silent resampling
- NO feature engineering
"""

import pandas as pd

from src.ingestion.tennet_common import (
    download_chunks,
    fmt_tennet_local,
    month_ranges,
    parse_amsterdam_to_utc,
    save_raw_table,
)


TABLE_NAME = "raw_tennet_settlement_prices"
SOURCE_NAME = "tennet_settlement_prices"
ENDPOINT = "/publications/v1/settlement-prices"

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
        ranges=month_ranges(START_DATE, END_DATE),
        parser_func=parse_payload,
        date_formatter=fmt_tennet_local,
        sleep_sec=1.2,
    )


def parse_payload(payload: dict) -> pd.DataFrame:
    response = payload.get("Response", {})
    timeseries_list = response.get("TimeSeries", [])

    rows = []

    for ts in timeseries_list:
        price_unit = ts.get("price_Measurement_Unit_name")
        currency = ts.get("currency_Unit_name")

        period = ts.get("Period", {})
        points = period.get("Points", [])

        for point in points:
            rows.append(
                {
                    "time_start": point.get("timeInterval_start"),
                    "time_end": point.get("timeInterval_end"),
                    "isp": point.get("isp"),
                    "incident_reserve_up": point.get("incident_reserve_up"),
                    "incident_reserve_down": point.get("incident_reserve_down"),
                    "dispatch_up": point.get("dispatch_up"),
                    "dispatch_down": point.get("dispatch_down"),
                    "shortage_price": point.get("shortage"),
                    "surplus_price": point.get("surplus"),
                    "regulation_state": point.get("regulation_state"),
                    "regulating_condition": point.get("regulating_condition"),
                    "price_unit": price_unit,
                    "currency": currency,
                }
            )

    return pd.DataFrame(rows)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = parse_amsterdam_to_utc(df["time_start"])
    df["time_end"] = parse_amsterdam_to_utc(df["time_end"])

    numeric_cols = [
        "isp",
        "incident_reserve_up",
        "incident_reserve_down",
        "dispatch_up",
        "dispatch_down",
        "shortage_price",
        "surplus_price",
        "regulation_state",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["time_start"])

    return df.set_index("timestamp").sort_index()


if __name__ == "__main__":
    main()
