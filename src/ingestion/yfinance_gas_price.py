import os
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import (
    GAS_TICKER,
    START_DATE_HISTORY,
    END_DATE_HISTORY_EXCLUSIVE,
    BASE_FREQ,
)


TABLE_NAME = "raw_gas_price_15min"


def main() -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    engine = create_engine(database_url)

    print("Downloading TTF gas prices...")

    df = yf.download(
        GAS_TICKER,
        start=START_DATE_HISTORY,
        end=END_DATE_HISTORY_EXCLUSIVE,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No gas price data downloaded")

    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["timestamp", "gas_price"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    df["gas_price_log"] = np.log1p(df["gas_price"])

    df_15min = df.resample(BASE_FREQ).ffill().reset_index()

    df_15min["source"] = "yfinance"
    df_15min["ticker"] = GAS_TICKER
    df_15min["created_at"] = pd.Timestamp.now(tz="UTC")

    df_15min.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
    )

    print("Gas price: OK")
    print(f"Rows: {len(df_15min)}")
    print(f"Range: {df_15min['timestamp'].min()} → {df_15min['timestamp'].max()}")
    print(f"Saved to table: {TABLE_NAME}")


if __name__ == "__main__":
    main()
