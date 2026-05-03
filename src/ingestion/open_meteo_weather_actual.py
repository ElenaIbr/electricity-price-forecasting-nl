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


TABLE_NAME = "raw_weather_actual_hourly"


def main() -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    engine = create_engine(database_url)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE_HISTORY,
        "end_date": END_DATE_HISTORY,
        "hourly": ",".join([
            "temperature_2m",
            "wind_speed_10m",
            "shortwave_radiation",
            "cloud_cover",
            "precipitation",
            "relative_humidity_2m",
        ]),
        "timezone": TIMEZONE,
    }

    print("Requesting Open-Meteo archive weather...")

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"No 'hourly' field in response. Keys: {list(data.keys())}")

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "temperature_c": hourly.get("temperature_2m"),
        "wind_ms": hourly.get("wind_speed_10m"),
        "solar_radiation": hourly.get("shortwave_radiation"),
        "cloud_cover": hourly.get("cloud_cover"),
        "precipitation": hourly.get("precipitation"),
        "humidity": hourly.get("relative_humidity_2m"),
    })

    df = df.set_index("timestamp").sort_index()
    df = df.resample(WEATHER_FREQ).mean().reset_index()

    df["source"] = "open_meteo_archive"
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
