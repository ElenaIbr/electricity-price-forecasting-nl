"""ENTSO-E wind + solar generation forecast (D+1) — operational.

Один query_wind_and_solar_forecast возвращает обе колонки.
Записываем как single wide table:
    timestamp UTC | country_code | wind_forecast_mw | solar_forecast_mw
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from entsoe import EntsoePandasClient

from src.config.settings import COUNTRY_CODE
from src.ingestion.base import BaseFetcher, OperationalWindow

logger = logging.getLogger(__name__)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ENTSO-E может вернуть MultiIndex columns. Сводим к плоским именам."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
    return df


def _pick_wind_solar(df: pd.DataFrame) -> pd.DataFrame:
    """Из произвольного набора wind/solar колонок собирает 2 числовые серии."""
    df = _flatten_columns(df)

    wind_cols  = [c for c in df.columns if "wind" in c]
    solar_cols = [c for c in df.columns if "solar" in c]

    out = pd.DataFrame(index=df.index)
    out["wind_forecast_mw"]  = df[wind_cols].sum(axis=1)  if wind_cols  else pd.NA
    out["solar_forecast_mw"] = df[solar_cols].sum(axis=1) if solar_cols else pd.NA
    return out


class EntsoeGenerationForecastOperational(BaseFetcher):
    table_name = "op_generation_forecast_15min"
    extra_filter_sql = '"country_code" = :cc'

    def __init__(self, country_code: str = COUNTRY_CODE) -> None:
        self.country_code = country_code
        api_token = os.getenv("ENTSOE_API_TOKEN")
        if not api_token:
            raise RuntimeError("ENTSOE_API_TOKEN is not set")
        self.client = EntsoePandasClient(api_key=api_token)

    def extra_delete_params(self):
        return {"cc": self.country_code}

    def fetch(self, window: OperationalWindow) -> pd.DataFrame:
        start_cet, end_cet = window.to_cet()
        logger.info(
            "GenerationForecast [%s] window %s … %s",
            self.country_code, start_cet, end_cet,
        )
        try:
            raw = self.client.query_wind_and_solar_forecast(
                country_code=self.country_code,
                start=start_cet,
                end=end_cet,
            )
        except Exception as exc:
            logger.warning("GenerationForecast query failed: %s", exc)
            return pd.DataFrame()

        if raw is None or len(raw) == 0:
            return pd.DataFrame()

        df = _pick_wind_solar(raw)

        if df.index.tz is None:
            df.index = df.index.tz_localize("Europe/Amsterdam")
        df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.reset_index()

        df["country_code"] = self.country_code
        df["source"] = "entsoe"
        df["fetched_at"] = pd.Timestamp.now(tz="UTC")
        return df


def main() -> None:
    from dotenv import load_dotenv
    from src.db.connection import get_engine

    load_dotenv()
    engine = get_engine()
    window = OperationalWindow.from_as_of()
    EntsoeGenerationForecastOperational().run(engine, window)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
