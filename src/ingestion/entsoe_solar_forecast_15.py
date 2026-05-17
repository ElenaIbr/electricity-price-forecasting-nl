import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import COUNTRY_CODE, LOCAL_TIMEZONE

START_YEAR = 2019
END_YEAR = 2025

TABLE_NAME = "raw_solar_forecast_15min"


def extract_solar_series(forecast: pd.DataFrame, debug: bool = False) -> pd.Series:
    if not isinstance(forecast, pd.DataFrame):
        raise TypeError(f"Unexpected forecast type: {type(forecast)}")

    if isinstance(forecast.columns, pd.MultiIndex):
        level0 = forecast.columns.get_level_values(0).astype(str)
        solar_cols = [
            col for col, top in zip(forecast.columns, level0)
            if "solar" in top.lower()
        ]
        if not solar_cols:
            raise KeyError("Solar columns not found in MultiIndex forecast.")
        return forecast[solar_cols].sum(axis=1)

    solar_candidates = [
        col for col in forecast.columns
        if "solar" in str(col).lower()
    ]

    if not solar_candidates:
        raise KeyError(f"Solar column not found. Available columns: {list(forecast.columns)}")

    return forecast[solar_candidates].sum(axis=1)


def main() -> None:
    load_dotenv()

    engine = create_engine(os.getenv("DATABASE_URL"))
    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set")

    client = EntsoePandasClient(api_key=api_token)
    local_tz = timezone(LOCAL_TIMEZONE)

    dfs = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"ENTSOE solar forecast: downloading {year}...")

        start = pd.Timestamp(f"{year}-01-01", tz=local_tz)
        end = pd.Timestamp(f"{year + 1}-01-01", tz=local_tz)

        forecast = client.query_wind_and_solar_forecast(
            country_code=COUNTRY_CODE,
            start=start,
            end=end,
        )

        solar = extract_solar_series(forecast)
        df_year = solar.to_frame(name="solar_forecast_mw")

        if df_year.index.tz is None:
            df_year.index = df_year.index.tz_localize(LOCAL_TIMEZONE)

        df_year.index = df_year.index.tz_convert("UTC")
        df_year.index.name = "timestamp"
        dfs.append(df_year)

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.reset_index()
    df["source"] = "entsoe"
    df["country_code"] = COUNTRY_CODE
    df["created_at"] = pd.Timestamp.now(tz="UTC")

    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    print(f"Saved {len(df)} rows to {TABLE_NAME}")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
