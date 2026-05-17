"""Gas TTF (Yahoo Finance) — operational.

В operational режиме нужны последние ~lookback дней daily close.
Никакого ffill в 15-min — это работа feature-engineering слоя.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.config.settings import GAS_TICKER
from src.ingestion.base import BaseFetcher, OperationalWindow

logger = logging.getLogger(__name__)


class GasTtfDailyOperational(BaseFetcher):
    table_name = "op_gas_price_daily"
    extra_filter_sql = '"ticker" = :tk'

    def __init__(self, ticker: str = GAS_TICKER) -> None:
        self.ticker = ticker

    def extra_delete_params(self):
        return {"tk": self.ticker}

    def fetch(self, window: OperationalWindow) -> pd.DataFrame:
        # yfinance принимает строки YYYY-MM-DD; end exclusive
        start = window.history_start.date().isoformat()
        end = (window.forecast_end.date() + pd.Timedelta(days=1)).isoformat()
        logger.info("yfinance %s: %s … %s", self.ticker, start, end)

        try:
            df = yf.download(
                self.ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.warning("yfinance failed: %s", exc)
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning("yfinance returned empty for %s", self.ticker)
            return pd.DataFrame()

        # Может прийти MultiIndex columns (yfinance >=0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index()[["Date", "Close"]]
        df.columns = ["timestamp", "gas_price"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["gas_price_log"] = np.log1p(df["gas_price"])
        df["ticker"] = self.ticker
        df["source"] = "yfinance"
        df["fetched_at"] = pd.Timestamp.now(tz="UTC")
        return df


def main() -> None:
    from dotenv import load_dotenv
    from src.db.connection import get_engine

    load_dotenv()
    engine = get_engine()
    window = OperationalWindow.from_as_of()
    GasTtfDailyOperational().run(engine, window)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
