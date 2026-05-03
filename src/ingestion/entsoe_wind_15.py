import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import COUNTRY_CODE, LOCAL_TIMEZONE

START_YEAR = 2023
END_YEAR = 2025
PSR_WIND = "B19"

TABLE_NAME = "raw_wind_generation_15min"


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
        print(f"ENTSOE wind generation: downloading {year}...")

        start = pd.Timestamp(f"{year}-01-01", tz=local_tz)
        end = pd.Timestamp(f"{year + 1}-01-01", tz=local_tz)

        wind = client.query_generation(
            country_code=COUNTRY_CODE,
            start=start,
            end=end,
            psr_type=PSR_WIND,
        )

        if isinstance(wind, pd.Series):
            df_year = wind.to_frame(name="wind_mw")
        elif isinstance(wind, pd.DataFrame):
            df_year = wind.sum(axis=1).to_frame(name="wind_mw")
        else:
            raise TypeError(f"Unexpected wind type: {type(wind)}")

        df_year.index = df_year.index.tz_convert("UTC")
        df_year.index.name = "timestamp"
        dfs.append(df_year)

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.reset_index()
    df["source"] = "entsoe"
    df["country_code"] = COUNTRY_CODE
    df["psr_type"] = PSR_WIND
    df["created_at"] = pd.Timestamp.now(tz="UTC")

    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    print(f"Saved {len(df)} rows to {TABLE_NAME}")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
