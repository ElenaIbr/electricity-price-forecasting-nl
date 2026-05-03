import os
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import (
    CO2_TICKER,
    START_DATE_HISTORY,
    END_DATE_HISTORY,
    BASE_FREQ,
)


TABLE_NAME = "raw_co2_price_15min"


def main() -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    engine = create_engine(database_url)

    print("Downloading CO2 price...")

    df = yf.download(
        CO2_TICKER,
        start=START_DATE_HISTORY,
        end=END_DATE_HISTORY,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No CO2 data")

    # ---- Normalize ----
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["timestamp", "co2_price"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # =========================
    # Resample to BASE_FREQ (15min)
    # =========================
    df_15min = df.resample(BASE_FREQ).ffill()

    # ---- Prepare for DB ----
    df_15min = df_15min.reset_index()

    df_15min["source"] = "yfinance"
    df_15min["ticker"] = CO2_TICKER
    df_15min["created_at"] = pd.Timestamp.now(tz="UTC")

    # ---- Save to PostgreSQL ----
    df_15min.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
    )

    print("CO2 price: OK")
    print(f"Rows: {len(df_15min)}")
    print(f"Range: {df_15min['timestamp'].min()} → {df_15min['timestamp'].max()}")
    print(f"Saved to table: {TABLE_NAME}")


if __name__ == "__main__":
    main()
