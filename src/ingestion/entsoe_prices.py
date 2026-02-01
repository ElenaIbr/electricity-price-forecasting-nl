# ================================
# Download and save Day-Ahead hourly electricity prices (NL)
# Source: ENTSO-E Transparency Platform
# Purpose: RAW data ingestion for diploma project
# ================================

from pathlib import Path
import os

import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone


# ---- Configuration ----
COUNTRY_CODE = "NL"
LOCAL_TZ = timezone("Europe/Amsterdam")

START_DATE = pd.Timestamp("2019-01-01", tz=LOCAL_TZ)
END_DATE   = pd.Timestamp("2026-01-01", tz=LOCAL_TZ)

OUTPUT_FILENAME = "nl_day_ahead_prices_hourly_2019_2025.csv"


def main() -> None:
    # ---- Project paths ----
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw" / "entsoe"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_FILENAME

    # ---- API token ----
    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set")

    client = EntsoePandasClient(api_key=api_token)

    # ---- Download ----
    prices = client.query_day_ahead_prices(
        COUNTRY_CODE,
        start=START_DATE,
        end=END_DATE
    )

    # ---- To DataFrame ----
    df = prices.to_frame(name="day_ahead_price")

    # ---- Time handling ----
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df.sort_index()

    # ---- Save RAW ----
    df.to_csv(output_path)

    print(f"Saved RAW hourly day-ahead prices to:\n{output_path}")
    print(f"Observations: {len(df)}")
    print(f"Range: {df.index.min()} → {df.index.max()}")


if __name__ == "__main__":
    main()
