"""Open-Meteo weather — operational.

В отличие от исторических скриптов, тут используется **forecast endpoint**
(api.open-meteo.com/v1/forecast), который возвращает:
  • past_days     — последние N дней realised weather
  • forecast_days — прогноз на следующие M дней

Это правильный endpoint для operational pipeline: один HTTP-вызов даёт
и недавний актуальный weather, и forecast на D+1.

Архивный endpoint (archive-api.open-meteo.com/v1/archive) имеет задержку
~5 дней и используется только в исторических backfill скриптах.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from src.config.settings import LATITUDE, LONGITUDE
from src.ingestion.base import BaseFetcher, OperationalWindow

logger = logging.getLogger(__name__)


HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "relative_humidity_2m",
    "wind_gusts_10m",
    "surface_pressure",
    "dew_point_2m",
]

VAR_RENAME = {
    "temperature_2m":         "temperature_c",
    "wind_speed_10m":         "wind_ms",
    "shortwave_radiation":    "solar_radiation",
    "cloud_cover":            "cloud_cover",
    "precipitation":          "precipitation",
    "relative_humidity_2m":   "humidity",
    "wind_gusts_10m":         "wind_gusts_ms",
    "surface_pressure":       "surface_pressure",
    "dew_point_2m":           "dew_point_c",
}


class OpenMeteoWeatherOperational(BaseFetcher):
    """Один общий fetcher: past_days actual + forecast_days forecast.

    Записываем в одну таблицу `op_weather_hourly` с колонкой `kind`:
      kind='actual'   — past_days часть (timestamp < as_of)
      kind='forecast' — будущие часы (timestamp >= as_of)
    Так и фича-инжиниринг сможет различать.
    """

    table_name = "op_weather_hourly"
    extra_filter_sql = '"latitude" = :lat AND "longitude" = :lon'

    URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        latitude: float = LATITUDE,
        longitude: float = LONGITUDE,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude

    def extra_delete_params(self):
        return {"lat": self.latitude, "lon": self.longitude}

    def fetch(self, window: OperationalWindow) -> pd.DataFrame:
        as_of_utc = window.as_of
        past_days = max(
            1,
            int((as_of_utc - window.history_start).total_seconds() // 86400) + 1,
        )
        forecast_days = max(
            1,
            int((window.forecast_end - as_of_utc).total_seconds() // 86400) + 1,
        )
        # Open-Meteo limits: past_days <= 92, forecast_days <= 16
        past_days = min(past_days, 92)
        forecast_days = min(forecast_days, 16)

        params = {
            "latitude":  self.latitude,
            "longitude": self.longitude,
            "hourly":    ",".join(HOURLY_VARS),
            "timezone":  "UTC",
            "past_days": past_days,
            "forecast_days": forecast_days,
        }

        logger.info(
            "Open-Meteo: past_days=%d forecast_days=%d (lat=%s, lon=%s)",
            past_days, forecast_days, self.latitude, self.longitude,
        )

        try:
            response = requests.get(self.URL, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("Open-Meteo request failed: %s", exc)
            return pd.DataFrame()

        if "hourly" not in data or not data["hourly"].get("time"):
            logger.warning("Open-Meteo: empty hourly payload, keys=%s", list(data.keys()))
            return pd.DataFrame()

        hourly = data["hourly"]
        df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True)})
        for src_col, dst_col in VAR_RENAME.items():
            df[dst_col] = hourly.get(src_col)

        df = df.sort_values("timestamp")
        df["kind"] = (df["timestamp"] >= as_of_utc).map({True: "forecast", False: "actual"})
        df["latitude"] = self.latitude
        df["longitude"] = self.longitude
        df["source"] = "open_meteo_forecast_api"
        df["fetched_at"] = pd.Timestamp.now(tz="UTC")
        return df


def main() -> None:
    from dotenv import load_dotenv
    from src.db.connection import get_engine

    load_dotenv()
    engine = get_engine()
    window = OperationalWindow.from_as_of()
    OpenMeteoWeatherOperational().run(engine, window)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
