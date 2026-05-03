import os
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    START_DATE_HISTORY,
    END_DATE_HISTORY,
    WEATHER_FREQ,
)


TABLE_NAME = "raw_weather_forecast_hourly"


def main() -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    engine = create_engine(database_url)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    hourly_vars = [
        "temperature_2m",
        "wind_speed_10m",
        "cloud_cover",
        "precipitation",
        "shortwave_radiation",
        "surface_pressure",
        "wind_gusts_10m",
        "dew_point_2m",
        "relative_humidity_2m",
    ]

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE_HISTORY,
        "end_date": END_DATE_HISTORY,
        "hourly": ",".join(hourly_vars),
        "timezone": TIMEZONE,
    }

    print("Requesting Open-Meteo Historical Forecast API...")

    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"No 'hourly' field in response. Keys: {list(data.keys())}")

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),

        "target_temp_forecast": hourly.get("temperature_2m"),
        "target_wind_speed_forecast": hourly.get("wind_speed_10m"),
        "target_cloud_cover_forecast": hourly.get("cloud_cover"),
        "target_precip_forecast": hourly.get("precipitation"),
        "target_shortwave_radiation_forecast": hourly.get("shortwave_radiation"),

        "target_surface_pressure_forecast": hourly.get("surface_pressure"),
        "target_wind_gusts_forecast": hourly.get("wind_gusts_10m"),
        "target_dew_point_forecast": hourly.get("dew_point_2m"),
        "target_relative_humidity_forecast": hourly.get("relative_humidity_2m"),
    })

    df = df.set_index("timestamp").sort_index()
    df = df.resample(WEATHER_FREQ).mean().reset_index()

    df["source"] = "open_meteo_historical_forecast"
    df["latitude"] = LATITUDE
    df["longitude"] = LONGITUDE
    df["created_at"] = pd.Timestamp.now(tz="UTC")

    print("Rows:", len(df))
    print("Range:", df["timestamp"].min(), "→", df["timestamp"].max())
    print("Missing values:")
    print(df.isna().sum())

    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
    )

    print(f"Saved to table: {TABLE_NAME}")


if __name__ == "__main__":
    main()
