# ================================
# Download HOURLY WIND generation (NL)
# Source: ENTSO-E Transparency Platform
# Period: 2019-01-01 → 2025-12-31
# Purpose: RAW wind generation data ingestion
# ================================

from pathlib import Path
import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone


COUNTRY_CODE = "NL"
LOCAL_TZ = timezone("Europe/Amsterdam")

START_YEAR = 2019
END_YEAR = 2025

PSR_WIND = "B19"  # Wind (onshore + offshore)
OUTPUT_FILENAME = "nl_wind_generation_hourly_2019_2025.csv"


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

    dfs: list[pd.DataFrame] = []

    # ---- Download by year ----
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"ENTSOE wind: downloading {year}…")

        start = pd.Timestamp(f"{year}-01-01", tz=LOCAL_TZ)
        end = pd.Timestamp(f"{year + 1}-01-01", tz=LOCAL_TZ)

        wind = client.query_generation(
            country_code=COUNTRY_CODE,
            start=start,
            end=end,
            psr_type=PSR_WIND,
        )

        # ---- Normalize output ----
        if isinstance(wind, pd.Series):
            df_year = wind.to_frame(name="wind_mw")

        elif isinstance(wind, pd.DataFrame):
            # Sum all wind subtypes defensively
            df_year = wind.sum(axis=1).to_frame(name="wind_mw")

        else:
            raise TypeError(f"Unexpected wind type: {type(wind)}")

        # ---- Time handling ----
        df_year.index = df_year.index.tz_convert("UTC")
        df_year.index.name = "timestamp"

        dfs.append(df_year)

    # ---- Combine all years ----
    df = pd.concat(dfs).sort_index()

    # ---- Save RAW ----
    df.to_csv(output_path)

    print("ENTSOE wind: OK")
    print(f"Rows: {len(df)} | Range: {df.index.min()} → {df.index.max()}")


if __name__ == "__main__":
    main()
