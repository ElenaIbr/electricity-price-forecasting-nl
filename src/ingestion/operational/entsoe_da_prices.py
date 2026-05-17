"""ENTSO-E day-ahead prices — operational fetcher.

Тянет hourly DA prices для NL/DE_LU/BE/FR в одной long-форме.
В operational режиме доступно:
  • история ДО as_of  (для лагов и rolling)
  • D     (если запуск >= 12:42 CET D-1, цены уже опубликованы)
  • D+1   (только если auction уже clear-ed)

Если D+1 ещё не опубликовано (запуск ДО 12:42 CET D-1), таблица просто
не получит этих часов — фича-инжиниринг должен использовать только лаги
для cross-border.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

import pandas as pd
from entsoe import EntsoePandasClient

from src.ingestion.base import BaseFetcher, OperationalWindow, to_utc_indexed

logger = logging.getLogger(__name__)


COUNTRIES: dict[str, str] = {
    "NL": "nl",
    "DE_LU": "de",
    "BE": "be",
    "FR": "fr",
}


class EntsoeDayAheadPricesOperational(BaseFetcher):
    table_name = "op_da_prices_hourly"
    time_column = "timestamp"

    def __init__(self, countries: Iterable[str] = None) -> None:
        self.countries = dict(COUNTRIES) if countries is None else {
            c: COUNTRIES[c] for c in countries
        }
        api_token = os.getenv("ENTSOE_API_TOKEN")
        if not api_token:
            raise RuntimeError("ENTSOE_API_TOKEN is not set")
        self.client = EntsoePandasClient(api_key=api_token)

    def _fetch_country(
        self,
        country_code: str,
        country_label: str,
        start_cet: pd.Timestamp,
        end_cet: pd.Timestamp,
    ) -> pd.DataFrame:
        logger.info("DA prices: %s [%s … %s]", country_code, start_cet, end_cet)
        try:
            raw = self.client.query_day_ahead_prices(
                country_code=country_code,
                start=start_cet,
                end=end_cet,
            )
        except Exception as exc:
            logger.warning("DA prices: %s failed (%s)", country_code, exc)
            return pd.DataFrame()

        if raw is None or len(raw) == 0:
            return pd.DataFrame()

        df = to_utc_indexed(raw, value_name="price_eur_mwh")
        # На всякий случай гарантируем hourly:
        df = df.resample("1h").mean().dropna(how="all")
        df = df.reset_index()
        df["country_code"] = country_code
        df["country_label"] = country_label
        return df

    def fetch(self, window: OperationalWindow) -> pd.DataFrame:
        start_cet, end_cet = window.to_cet()
        frames: list[pd.DataFrame] = []
        for country_code, country_label in self.countries.items():
            df = self._fetch_country(country_code, country_label, start_cet, end_cet)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        out["source"] = "entsoe"
        out["frequency"] = "hourly"
        out["fetched_at"] = pd.Timestamp.now(tz="UTC")
        out = out.sort_values(["country_code", "timestamp"]).reset_index(drop=True)
        return out


def main() -> None:
    """Удобство для одиночного запуска."""
    from dotenv import load_dotenv
    from src.db.connection import get_engine

    load_dotenv()
    engine = get_engine()
    window = OperationalWindow.from_as_of()
    EntsoeDayAheadPricesOperational().run(engine, window)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
