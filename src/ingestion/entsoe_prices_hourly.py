import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import (
    ENTSOE_START_DATE,
    ENTSOE_END_DATE,
    LOCAL_TIMEZONE,
)


COUNTRIES = {
    "NL": "nl",
    "DE_LU": "de",
    "BE": "be",
    "FR": "fr",
}


def load_day_ahead_price(
    client: EntsoePandasClient,
    country_code: str,
    country_label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    print(f"\nENTSOE day-ahead prices: downloading {country_code}...")

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
            raise ValueError(
                f"Expected one price column for {country_code}, "
                f"got {prices.shape[1]}"
            )
        df = prices.copy()
        df.columns = [column_name]
    else:
        raise TypeError(f"Unexpected prices type for {country_code}: {type(prices)}")

    if df.index.tz is None:
        df.index = df.index.tz_localize(LOCAL_TIMEZONE)

    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    print("Original time step distribution:")
    print(df.index.to_series().diff().value_counts(dropna=True).head())

    # Force hourly because output table is explicitly *_hourly.
    # If data is already hourly, this does not change it.
    df = (
        df
        .resample("1h")
        .mean()
        .dropna(how="all")
    )

    print("After hourly aggregation:")
    print(df.index.to_series().diff().value_counts(dropna=True).head())

    df = df.reset_index()

    df["source"] = "entsoe"
    df["country_code"] = country_code
    df["country_label"] = country_label
    df["frequency"] = "hourly"
    df["frequency_note"] = "aggregated_to_hourly_if_needed"
    df["created_at"] = pd.Timestamp.now(tz="UTC")

    return df


def save_country_price(
    engine,
    country_code: str,
    country_label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: EntsoePandasClient,
) -> None:
    table_name = f"raw_{country_label}_day_ahead_price_hourly"

    df = load_day_ahead_price(
        client=client,
        country_code=country_code,
        country_label=country_label,
        start=start,
        end=end,
    )

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
    )

    print(f"Saved {len(df)} rows to {table_name}")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")


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

    start = pd.Timestamp(ENTSOE_START_DATE, tz=local_tz)
    end = pd.Timestamp(ENTSOE_END_DATE, tz=local_tz)

    for country_code, country_label in COUNTRIES.items():
        save_country_price(
            engine=engine,
            client=client,
            country_code=country_code,
            country_label=country_label,
            start=start,
            end=end,
        )


if __name__ == "__main__":
    main()
