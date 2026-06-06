from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine
from src.features.build_features import FeatureContext, build_feature_frame
from src.features.feature_spec import TARGET
from src.models.pipeline import ForecastPipeline

logger = logging.getLogger(__name__)


FORECAST_TABLE = "op_forecasts_hourly"


def read_master_frame_from_db(
    as_of: pd.Timestamp,
    lookback_days: int = 60,
    forecast_days: int = 2,
) -> pd.DataFrame:
    """
    Reads operational data from PostgreSQL and builds the hourly master frame
    required by build_feature_frame().
    """
    engine = get_engine()

    start = (as_of - pd.Timedelta(days=lookback_days)).floor("D")
    end = (as_of + pd.Timedelta(days=forecast_days)).ceil("D")

    # DA prices: NL target + cross-border prices
    da = pd.read_sql(
        text("""
            SELECT timestamp, country_label, price_eur_mwh
            FROM op_da_prices_hourly
            WHERE timestamp >= :start AND timestamp <= :end
        """),
        engine,
        params={"start": start, "end": end},
    )

    da["timestamp"] = pd.to_datetime(da["timestamp"], utc=True)

    da_wide = (
        da.pivot_table(
            index="timestamp",
            columns="country_label",
            values="price_eur_mwh",
            aggfunc="last",
        )
        .rename(columns={
            "nl": "nl_day_ahead_price",
            "be": "be_day_ahead_price",
            "de": "de_day_ahead_price",
            "fr": "fr_day_ahead_price",
        })
    )

    # Load forecast
    load_fc = pd.read_sql(
        text("""
            SELECT timestamp, load_forecast_mw
            FROM op_load_forecast_15min
            WHERE timestamp >= :start AND timestamp <= :end
        """),
        engine,
        params={"start": start, "end": end},
    )
    load_fc["timestamp"] = pd.to_datetime(load_fc["timestamp"], utc=True)
    load_fc = (
        load_fc.set_index("timestamp")
        .resample("1h")
        .mean()
        .rename(columns={"load_forecast_mw": "load_forecast"})
    )

    # Generation forecast
    gen_fc = pd.read_sql(
        text("""
            SELECT timestamp, wind_forecast_mw, solar_forecast_mw
            FROM op_generation_forecast_15min
            WHERE timestamp >= :start AND timestamp <= :end
        """),
        engine,
        params={"start": start, "end": end},
    )
    gen_fc["timestamp"] = pd.to_datetime(gen_fc["timestamp"], utc=True)
    gen_fc = gen_fc.set_index("timestamp").resample("1h").mean()

    # Weather actual + forecast from one operational table
    weather = pd.read_sql(
        text("""
            SELECT
                timestamp,
                kind,
                temperature_c,
                wind_ms,
                solar_radiation,
                cloud_cover,
                humidity
            FROM op_weather_hourly
            WHERE timestamp >= :start AND timestamp <= :end
        """),
        engine,
        params={"start": start, "end": end},
    )
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)

    weather_actual = (
        weather[weather["kind"] == "actual"]
        .set_index("timestamp")
        [["temperature_c", "wind_ms", "solar_radiation", "cloud_cover", "humidity"]]
        .resample("1h")
        .mean()
    )

    weather_forecast = (
        weather[weather["kind"] == "forecast"]
        .set_index("timestamp")
        [["temperature_c", "wind_ms", "solar_radiation"]]
        .resample("1h")
        .mean()
        .rename(columns={
            "temperature_c": "temperature_forecast",
            "wind_ms": "wind_speed_forecast",
            "solar_radiation": "solar_radiation_forecast",
        })
    )

    # Gas daily, forward-filled to hourly
    gas = pd.read_sql(
        text("""
            SELECT timestamp, gas_price
            FROM op_gas_price_daily
            WHERE timestamp >= :start AND timestamp <= :end
        """),
        engine,
        params={"start": start - pd.Timedelta(days=3), "end": end},
    )
    gas["timestamp"] = pd.to_datetime(gas["timestamp"], utc=True)
    gas = gas.set_index("timestamp").sort_index()
    gas_hourly = gas.resample("1h").ffill()

    hourly_index = pd.date_range(start=start, end=end, freq="1h", tz="UTC")

    master = pd.DataFrame(index=hourly_index)
    for part in [da_wide, load_fc, gen_fc, weather_actual, weather_forecast, gas_hourly]:
        master = master.join(part, how="left")

    # Temporary placeholders until these operational sources are wired.
    # Better than pretending the universe gave us data it did not.
    master["net_flow_de_nl"] = 0.0
    master["net_flow_be_nl"] = 0.0
    master["imbalance_price_long"] = master[TARGET].shift(24)
    master["imbalance_price_short"] = master[TARGET].shift(24)

    master = master.sort_index()
    return master


def save_forecast_to_db(
    y_pred: pd.Series,
    target_date: date,
    model_version: str,
    as_of: pd.Timestamp,
) -> None:
    engine = get_engine()

    df = pd.DataFrame({
        "timestamp": y_pred.index,
        "target_date": pd.Timestamp(target_date).date(),
        "predicted_price": y_pred.values,
        "model_version": model_version,
        "as_of": as_of,
        "created_at": pd.Timestamp.now(tz="UTC"),
    })

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                DELETE FROM {FORECAST_TABLE}
                WHERE target_date = :target_date
                  AND model_version = :model_version
            """),
            {
                "target_date": pd.Timestamp(target_date).date(),
                "model_version": model_version,
            },
        )
        df.to_sql(FORECAST_TABLE, conn, if_exists="append", index=False)


def run_daily_pipeline(
    as_of: pd.Timestamp | None = None,
    target_date: date | None = None,
) -> pd.Series:
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC").floor("h")
    else:
        as_of = pd.Timestamp(as_of)
        as_of = as_of.tz_localize("UTC") if as_of.tz is None else as_of.tz_convert("UTC")

    if target_date is None:
        target_date = (as_of.tz_convert("Europe/Amsterdam") + pd.Timedelta(days=1)).date()

    logger.info("Daily pipeline: as_of=%s target_date=%s", as_of, target_date)

    pipe = ForecastPipeline.from_bundle("current")

    master = read_master_frame_from_db(as_of=as_of)

    ctx = FeatureContext(
        as_of=as_of,
        target_date=pd.Timestamp(target_date, tz="UTC"),
    )

    feats = build_feature_frame(master, pipe.bundle.feature_params, ctx)

    day_start = pd.Timestamp(target_date, tz="Europe/Amsterdam").tz_convert("UTC")
    day_end = (pd.Timestamp(target_date, tz="Europe/Amsterdam") + pd.Timedelta(days=1)).tz_convert("UTC")

    target_rows = feats.loc[day_start:day_end - pd.Timedelta(seconds=1)]

    if target_rows.empty:
        raise RuntimeError(f"No feature rows for target_date={target_date}")

    if target_rows.isna().any().any():
        cols = target_rows.columns[target_rows.isna().any()].tolist()
        raise RuntimeError(f"NaN in feature rows for {target_date}: {cols}")

    y_pred = pipe.predict(target_rows)
    y_pred.name = "predicted_price"

    save_forecast_to_db(
        y_pred=y_pred,
        target_date=target_date,
        model_version=pipe.bundle.version,
        as_of=as_of,
    )

    logger.info("Saved %d forecast rows into %s", len(y_pred), FORECAST_TABLE)
    return y_pred


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    y = run_daily_pipeline()
    print(y)


if __name__ == "__main__":
    main()
