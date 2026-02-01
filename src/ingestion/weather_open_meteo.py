# ================================
# Download hourly weather data from Open-Meteo (ERA5-based)
# Location: Netherlands (De Bilt area)
# Variables: temperature, wind speed
# Period: 2019-01-01 → 2025-12-31
# Purpose: RAW data ingestion
# ================================

from pathlib import Path
import requests
import pandas as pd


# ---- Location (De Bilt, NL) ----
LATITUDE = 52.10
LONGITUDE = 5.18

START_DATE = "2019-01-01"
END_DATE   = "2025-12-31"

OUTPUT_FILENAME = "weather_hourly_nl_open_meteo_2019_2025.csv"


def main() -> None:
    # ---- Project paths ----
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw" / "weather"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_FILENAME

    # ---- Open-Meteo Historical API ----
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m,wind_speed_10m",
        "timezone": "Europe/Amsterdam",
    }

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    # ---- Convert to DataFrame ----
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"],
        "wind_ms": data["hourly"]["wind_speed_10m"],
    })

    df = df.sort_values("timestamp")

    # ---- Save RAW ----
    df.to_csv(output_path, index=False)

    print("Weather (Open-Meteo): OK")
    print(f"Rows: {len(df)} | Range: {df.timestamp.min()} → {df.timestamp.max()}")


if __name__ == "__main__":
    main()
    