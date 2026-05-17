"""ENTSO-E load — actual & forecast (operational).

LoadActual    — реализованная нагрузка (доступна с задержкой ~1-2 часа)
LoadForecast  — прогноз TSO на D+1 (публикуется накануне, известен до auction)

Обе серии 15-min. В operational окне:
  • actual:    [as_of - lookback, as_of)        — известен факт
  • forecast:  [as_of - lookback, as_of + 2d]   — прогноз доступен на D+1
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from entsoe import EntsoePandasClient

from src.config.settings import COUNTRY_CODE
from src.ingestion.base import BaseFetcher, OperationalWindow, to_utc_indexed

logger = logging.getLogger(__name__)


class _EntsoeLoadBase(BaseFetcher):
    """Общая логика для actual / forecast load."""

    value_column: str = "load_mw"
    table_name: str = ""
    extra_filter_sql = '"country_code" = :cc'

    def __init__(self, country_code: str = COUNTRY_CODE) -> None:
        self.country_code = country_code
        api_token = os.getenv("ENTSOE_API_TOKEN")
        if not api_token:
            raise RuntimeError("ENTSOE_API_TOKEN is not set")
        self.client = EntsoePandasClient(api_key=api_token)

    def extra_delete_params(self):
        return {"cc": self.country_code}

    def _entsoe_call(self, start_cet, end_cet):
        raise NotImplementedError

    def fetch(self, window: OperationalWindow) -> pd.DataFrame:
        start_cet, end_cet = window.to_cet()
        logger.info(
            "%s [%s] window %s … %s",
            self.__class__.__name__, self.country_code, start_cet, end_cet,
        )
        try:
            raw = self._entsoe_call(start_cet, end_cet)
        except Exception as exc:
            logger.warning("%s: query failed (%s)", self.__class__.__name__, exc)
            return pd.DataFrame()

        if raw is None or len(raw) == 0:
            return pd.DataFrame()

        df = to_utc_indexed(raw, value_name=self.value_column)
        df = df.reset_index()
        df["country_code"] = self.country_code
        df["source"] = "entsoe"
        df["fetched_at"] = pd.Timestamp.now(tz="UTC")
        return df


class EntsoeLoadActualOperational(_EntsoeLoadBase):
    table_name = "op_load_actual_15min"
    value_column = "load_mw"

    def _entsoe_call(self, start_cet, end_cet):
        return self.client.query_load(
            country_code=self.country_code,
            start=start_cet,
            end=end_cet,
        )


class EntsoeLoadForecastOperational(_EntsoeLoadBase):
    table_name = "op_load_forecast_15min"
    value_column = "load_forecast_mw"

    def _entsoe_call(self, start_cet, end_cet):
        return self.client.query_load_forecast(
            country_code=self.country_code,
            start=start_cet,
            end=end_cet,
        )


def main() -> None:
    from dotenv import load_dotenv
    from src.db.connection import get_engine

    load_dotenv()
    engine = get_engine()
    window = OperationalWindow.from_as_of()
    EntsoeLoadActualOperational().run(engine, window)
    EntsoeLoadForecastOperational().run(engine, window)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
