"""
INGESTION SCRIPT: TenneT frequency restoration reserve activations

RAW table:
- raw_tennet_frr_activations

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

TABLE_NAME = "raw_tennet_frr_activations"
SOURCE_NAME = "tennet_frequency_restoration_reserve_activations"
ENDPOINT = "/publications/v1/frequency-restoration-reserve-activations"

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
        ranges=day_ranges(START_DATE, END_DATE),
        parser_func=parse_payload,
        date_formatter=fmt_tennet_local,
        sleep_sec=1.2,
    )


def parse_payload(payload: dict) -> pd.DataFrame:
    response = payload.get("Response", {})
    timeseries_list = response.get("TimeSeries", [])

    rows = []

    for ts in timeseries_list:
        quantity_unit = ts.get("quantity_Measurement_Unit_name")

        period = ts.get("Period", {})
        points = period.get("Points", [])

        for point in points:
            rows.append(
                {
                    "time_start": point.get("timeInterval_start"),
                    "time_end": point.get("timeInterval_end"),
                    "isp": point.get("isp"),
                    "afrr_down": point.get("aFRR_down"),
                    "afrr_up": point.get("aFRR_up"),
                    "mfrrda_volume_down": point.get("mfrrda_volume_down"),
                    "mfrrda_volume_up": point.get("mfrrda_volume_up"),
                    "absolute_total_volume": point.get("absolute_total_volume"),
                    "total_volume": point.get("total_volume"),
                    "quantity_unit": quantity_unit,
                }
            )

    return pd.DataFrame(rows)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = parse_amsterdam_to_utc(df["time_start"])
    df["time_end"] = parse_amsterdam_to_utc(df["time_end"])

    numeric_cols = [
        "isp",
        "afrr_down",
        "afrr_up",
        "mfrrda_volume_down",
        "mfrrda_volume_up",
        "absolute_total_volume",
        "total_volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["time_start"])

    return df.set_index("timestamp").sort_index()


if __name__ == "__main__":
    main()
