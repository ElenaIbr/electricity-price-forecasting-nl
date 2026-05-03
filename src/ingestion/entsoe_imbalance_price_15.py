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


TABLE_NAME = "raw_imbalance_price_15min"


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

    start = pd.Timestamp(START_DATE_HISTORY, tz=local_tz)
    end = pd.Timestamp(END_DATE_HISTORY_EXCLUSIVE, tz=local_tz)

    print("ENTSOE imbalance prices: downloading...")

    prices = client.query_imbalance_prices(
        COUNTRY_CODE,
        start=start,
        end=end,
    )

    if isinstance(prices, pd.Series):
        raise RuntimeError("Expected DataFrame with Long/Short columns")

    df = prices.copy()

    # ---- Normalize columns ----
    rename_map = {}

    for col in df.columns:
        col_str = str(col).strip().lower()

        if col_str == "long":
            rename_map[col] = "imbalance_price_long"
        elif col_str == "short":
            rename_map[col] = "imbalance_price_short"

    df = df.rename(columns=rename_map)

    expected_cols = [
        "imbalance_price_long",
        "imbalance_price_short",
    ]

    if not set(expected_cols).issubset(df.columns):
        raise RuntimeError(
            f"Unexpected columns returned: {list(df.columns)}"
        )

    df = df[expected_cols]

    # ---- Time handling ----
    if df.index.tz is None:
        df.index = df.index.tz_localize(LOCAL_TIMEZONE)

    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # ---- Check real frequency BEFORE any resampling ----
    step_counts = df.index.to_series().diff().value_counts(dropna=True)

    print("\nOriginal time step distribution:")
    print(step_counts.head(10))

    expected_step = pd.Timedelta(BASE_FREQ)

    if step_counts.empty:
        raise RuntimeError("Cannot infer time step: dataframe has too few rows")

    most_common_step = step_counts.index[0]

    if most_common_step != expected_step:
        raise RuntimeError(
            f"Expected {BASE_FREQ} imbalance prices, "
            f"but most common step is {most_common_step}"
        )

    # ---- Align to full 15-min grid WITHOUT hiding gaps ----
    expected_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=BASE_FREQ,
        tz="UTC",
    )

    df_15 = df.reindex(expected_index)
    df_15.index.name = "timestamp"

    missing = df_15.isna().sum()

    print("\nAfter reindex to full 15-min grid:")
    print("Rows:", len(df_15))
    print("Range:", df_15.index.min(), "→", df_15.index.max())
    print("Missing values:")
    print(missing)

    if missing.any():
        print("\nWarning: missing values found after reindex.")
        print("They are NOT forward-filled in this raw table.")

    # ---- Save to DB ----
    df_15 = df_15.reset_index()

    df_15["source"] = "entsoe"
    df_15["country_code"] = COUNTRY_CODE
    df_15["frequency"] = BASE_FREQ
    df_15["frequency_note"] = "native_15min_checked_no_ffill"
    df_15["created_at"] = pd.Timestamp.now(tz="UTC")

    df_15.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
    )

    print(f"\nSaved {len(df_15)} rows to {TABLE_NAME}")
    print(f"Range: {df_15['timestamp'].min()} → {df_15['timestamp'].max()}")


if __name__ == "__main__":
    main()
