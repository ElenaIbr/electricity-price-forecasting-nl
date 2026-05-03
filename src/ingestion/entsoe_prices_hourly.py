import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import ENTSOE_START_DATE, ENTSOE_END_DATE, LOCAL_TIMEZONE

TABLE_NAME = "raw_day_ahead_price_15min"


def main(
    country_code: str = "NL",
    country_label: str = "nl",
    start_date: str = ENTSOE_START_DATE,
    end_date: str = ENTSOE_END_DATE,
) -> None:
    load_dotenv()

    engine = create_engine(os.getenv("DATABASE_URL"))
    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set")

    client = EntsoePandasClient(api_key=api_token)
    local_tz = timezone(LOCAL_TIMEZONE)

    start = pd.Timestamp(start_date, tz=local_tz)
    end = pd.Timestamp(end_date, tz=local_tz)

    print(f"ENTSOE day-ahead prices: downloading {country_code}...")

    prices = client.query_day_ahead_prices(
        country_code=country_code,
        start=start,
        end=end,
    )

    column_name = f"{country_label}_day_ahead_price"

    if isinstance(prices, pd.Series):
        df = prices.to_frame(name=column_name)
    elif isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(f"Expected one price column, got {prices.shape[1]}")
        df = prices.copy()
        df.columns = [column_name]
    else:
        raise TypeError(f"Unexpected prices type: {type(prices)}")

    if df.index.tz is None:
        df.index = df.index.tz_localize(LOCAL_TIMEZONE)

    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.reset_index()
    df["source"] = "entsoe"
    df["country_code"] = country_code
    df["country_label"] = country_label
    df["created_at"] = pd.Timestamp.now(tz="UTC")

    table_name = f"raw_{country_label}_day_ahead_price_hourly"

    df.to_sql(table_name, engine, if_exists="replace", index=False)

    print(f"Saved {len(df)} rows to {table_name}")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
