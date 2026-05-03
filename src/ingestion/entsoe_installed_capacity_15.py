import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import (
    COUNTRY_CODE,
    LOCAL_TIMEZONE,
    START_DATE_HISTORY,
    END_DATE_HISTORY_EXCLUSIVE,
    BASE_FREQ,
)


TABLE_NAME = "raw_installed_capacity_15min"


def main() -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set")

    engine = create_engine(database_url)
    client = EntsoePandasClient(api_key=api_token)

    local_tz = timezone(LOCAL_TIMEZONE)

    start_year = pd.Timestamp(START_DATE_HISTORY).year
    end_year = pd.Timestamp(END_DATE_HISTORY_EXCLUSIVE).year - 1

    dfs: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        print(f"ENTSOE installed capacity: downloading {year}...")

        start = pd.Timestamp(f"{year}-01-01", tz=local_tz)
        end = pd.Timestamp(f"{year + 1}-01-01", tz=local_tz)

        data = client.query_installed_generation_capacity(
            COUNTRY_CODE,
            start=start,
            end=end,
        )

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Unexpected type: {type(data)}")

        df_year = data.copy()

        if isinstance(df_year.columns, pd.MultiIndex):
            df_year.columns = [
                "_".join([str(x) for x in col if x])
                for col in df_year.columns
            ]

        if df_year.index.tz is None:
            df_year.index = df_year.index.tz_localize(LOCAL_TIMEZONE)

        df_year.index = df_year.index.tz_convert("UTC")
        df_year.index.name = "timestamp"

        dfs.append(df_year)

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df_15 = df.resample(BASE_FREQ).ffill()

    df_15 = df_15.reset_index()
    df_15["source"] = "entsoe"
    df_15["country_code"] = COUNTRY_CODE
    df_15["frequency"] = BASE_FREQ
    df_15["frequency_note"] = "installed_capacity_forward_filled_to_15min"
    df_15["created_at"] = pd.Timestamp.now(tz="UTC")

    df_15.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
    )

    print(f"Saved {len(df_15)} rows to {TABLE_NAME}")
    print(f"Range: {df_15['timestamp'].min()} → {df_15['timestamp'].max()}")


if __name__ == "__main__":
    main()
