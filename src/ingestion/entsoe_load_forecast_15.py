import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import COUNTRY_CODE, ENTSOE_START_DATE, ENTSOE_END_DATE, LOCAL_TIMEZONE

TABLE_NAME = "raw_load_forecast_15min"


def main() -> None:
    load_dotenv()

    engine = create_engine(os.getenv("DATABASE_URL"))
    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set")

    client = EntsoePandasClient(api_key=api_token)
    local_tz = timezone(LOCAL_TIMEZONE)

    start = pd.Timestamp(ENTSOE_START_DATE, tz=local_tz)
    end = pd.Timestamp(ENTSOE_END_DATE, tz=local_tz)

    print("ENTSOE load forecast: downloading...")

    forecast = client.query_load_forecast(
        COUNTRY_CODE,
        start=start,
        end=end,
    )

    if isinstance(forecast, pd.Series):
        df = forecast.to_frame(name="load_forecast")
    elif isinstance(forecast, pd.DataFrame):
        if forecast.shape[1] == 1:
            df = forecast.copy()
            df.columns = ["load_forecast"]
        else:
            df = forecast.iloc[:, 0].to_frame(name="load_forecast")
    else:
        raise TypeError(f"Unexpected forecast type: {type(forecast)}")

    if df.index.tz is None:
        df.index = df.index.tz_localize(LOCAL_TIMEZONE)

    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = df.reset_index()
    df["source"] = "entsoe"
    df["country_code"] = COUNTRY_CODE
    df["created_at"] = pd.Timestamp.now(tz="UTC")

    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    print(f"Saved {len(df)} rows to {TABLE_NAME}")
    print(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
