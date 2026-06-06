"""POST /forecast — main inference endpoint.

Operational flow:

Take as_of / target_date from the request, or use defaults.
Read operational data from PostgreSQL.
Build the hourly master frame.
Run build_feature_frame(master, params, ctx).
Select the 24 hours of target_date.
Run pipeline.predict(X).
Return ForecastResponse.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.api.dependencies import PipelineDep
from src.api.schemas import (
    ForecastDebugResponse,
    ForecastRequest,
    ForecastResponse,
    HourlyDebug,
    HourlyForecast,
)
from src.db.connection import get_engine
from src.features.build_features import FeatureContext, build_feature_frame
from src.features.feature_spec import INPUT_COLUMNS, TARGET

logger = logging.getLogger(__name__)

router = APIRouter(tags=["forecast"], prefix="/forecast")

LOCAL_TZ = "Europe/Amsterdam"


def _ensure_utc(ts: datetime | pd.Timestamp | None) -> pd.Timestamp:
    if ts is None:
        return pd.Timestamp.now(tz="UTC").floor("h")

    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _default_target_date(as_of: pd.Timestamp) -> date:
    return (as_of.tz_convert(LOCAL_TZ) + pd.Timedelta(days=1)).date()


def _target_bounds_utc(target_date: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_local = pd.Timestamp(target_date, tz=LOCAL_TZ)
    end_local = start_local + pd.Timedelta(days=1)
    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def _read_sql_df(query: str, params: dict) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(text(query), engine, params=params)


def read_master_frame_from_db(
    as_of: pd.Timestamp,
    lookback_days: int = 60,
    forecast_days: int = 2,
) -> pd.DataFrame:
    start = (as_of - pd.Timedelta(days=lookback_days)).floor("D")
    end = (as_of + pd.Timedelta(days=forecast_days)).ceil("D")

    params = {"start": start, "end": end}

    da = _read_sql_df(
        """
        SELECT timestamp, country_label, price_eur_mwh
        FROM op_da_prices_hourly
        WHERE timestamp >= :start AND timestamp <= :end
        """,
        params,
    )

    if da.empty:
        raise HTTPException(503, "No day-ahead prices found in DB")

    da["timestamp"] = pd.to_datetime(da["timestamp"], utc=True)

    da_wide = (
        da.pivot_table(
            index="timestamp",
            columns="country_label",
            values="price_eur_mwh",
            aggfunc="last",
        )
        .rename(
            columns={
                "nl": "nl_day_ahead_price",
                "be": "be_day_ahead_price",
                "de": "de_day_ahead_price",
                "fr": "fr_day_ahead_price",
            }
        )
    )

    load_fc = _read_sql_df(
        """
        SELECT timestamp, load_forecast_mw
        FROM op_load_forecast_15min
        WHERE timestamp >= :start AND timestamp <= :end
        """,
        params,
    )
    load_fc["timestamp"] = pd.to_datetime(load_fc["timestamp"], utc=True)
    load_fc = (
        load_fc.set_index("timestamp")
        .resample("1h")
        .mean()
        .rename(columns={"load_forecast_mw": "load_forecast"})
    )

    gen_fc = _read_sql_df(
        """
        SELECT timestamp, wind_forecast_mw, solar_forecast_mw
        FROM op_generation_forecast_15min
        WHERE timestamp >= :start AND timestamp <= :end
        """,
        params,
    )
    gen_fc["timestamp"] = pd.to_datetime(gen_fc["timestamp"], utc=True)
    gen_fc = gen_fc.set_index("timestamp").resample("1h").mean()

    weather = _read_sql_df(
        """
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
        """,
        params,
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
        .rename(
            columns={
                "temperature_c": "temperature_forecast",
                "wind_ms": "wind_speed_forecast",
                "solar_radiation": "solar_radiation_forecast",
            }
        )
    )

    gas = _read_sql_df(
        """
        SELECT timestamp, gas_price
        FROM op_gas_price_daily
        WHERE timestamp >= :gas_start AND timestamp <= :end
        """,
        {"gas_start": start - pd.Timedelta(days=5), "end": end},
    )
    gas["timestamp"] = pd.to_datetime(gas["timestamp"], utc=True)
    gas_hourly = gas.set_index("timestamp").sort_index().resample("1h").ffill()

    hourly_index = pd.date_range(start=start, end=end, freq="1h", tz="UTC")
    master = pd.DataFrame(index=hourly_index)

    for part in [da_wide, load_fc, gen_fc, weather_actual, weather_forecast, gas_hourly]:
        master = master.join(part, how="left")

    # TODO: заменить на реальные operational tables.
    master["net_flow_de_nl"] = 0.0
    master["net_flow_be_nl"] = 0.0
    master["imbalance_price_long"] = master[TARGET].shift(24)
    master["imbalance_price_short"] = master[TARGET].shift(24)

    missing = [c for c in INPUT_COLUMNS if c not in master.columns]
    if missing:
        raise HTTPException(
            500,
            detail={
                "code": "missing_master_columns",
                "missing": missing,
            },
        )

    return master.sort_index()


def _slice_target_rows(feats: pd.DataFrame, target_date: date) -> pd.DataFrame:
    day_start_utc, day_end_utc = _target_bounds_utc(target_date)
    target_rows = feats.loc[day_start_utc : day_end_utc - pd.Timedelta(seconds=1)]

    if target_rows.empty:
        raise HTTPException(
            400,
            detail=f"No feature rows for target_date={target_date}",
        )

    if target_rows.isna().any().any():
        n_nan = int(target_rows.isna().sum().sum())
        cols = target_rows.columns[target_rows.isna().any()].tolist()
        raise HTTPException(
            422,
            detail={
                "code": "nan_in_features",
                "extra": {"n_nan": n_nan, "cols_with_nan": cols},
                "detail": (
                    "Computed features contain NaN. "
                    "Check DB coverage and forecast availability."
                ),
            },
        )

    return target_rows


def _build_target_rows_from_db(
    pipe,
    as_of: datetime | None,
    target_date: date | None,
    lookback_days: int,
    forecast_days: int,
) -> tuple[pd.DataFrame, date]:
    as_of_utc = _ensure_utc(as_of)
    target = target_date or _default_target_date(as_of_utc)

    master = read_master_frame_from_db(
        as_of=as_of_utc,
        lookback_days=lookback_days,
        forecast_days=forecast_days,
    )

    ctx = FeatureContext(
        as_of=as_of_utc,
        target_date=pd.Timestamp(target, tz="UTC"),
    )

    try:
        feats = build_feature_frame(master, pipe.bundle.feature_params, ctx)
    except Exception as exc:
        logger.exception("Feature engineering failed")
        raise HTTPException(500, detail=f"Feature engineering failed: {exc}")

    return _slice_target_rows(feats, target), target


@router.get("", response_model=ForecastResponse)
def forecast(
    pipe: PipelineDep,
    target_date: date | None = Query(default=None),
    as_of: datetime | None = Query(default=None),
    lookback_days: int = Query(default=60, ge=35, le=180),
    forecast_days: int = Query(default=2, ge=1, le=7),
) -> ForecastResponse:
    target_rows, target = _build_target_rows_from_db(
        pipe=pipe,
        as_of=as_of,
        target_date=target_date,
        lookback_days=lookback_days,
        forecast_days=forecast_days,
    )

    y = pipe.predict(target_rows)
    now = datetime.now(timezone.utc)

    return ForecastResponse(
        target_date=target,
        model_version=pipe.bundle.version,
        forecast_made_at=now,
        n_hours=len(y),
        hourly=[
            HourlyForecast(timestamp=ts.to_pydatetime(), predicted_price=float(v))
            for ts, v in y.items()
        ],
    )


@router.get("/debug", response_model=ForecastDebugResponse)
def forecast_debug(
    pipe: PipelineDep,
    target_date: date | None = Query(default=None),
    as_of: datetime | None = Query(default=None),
    lookback_days: int = Query(default=60, ge=35, le=180),
    forecast_days: int = Query(default=2, ge=1, le=7),
) -> ForecastDebugResponse:
    target_rows, target = _build_target_rows_from_db(
        pipe=pipe,
        as_of=as_of,
        target_date=target_date,
        lookback_days=lookback_days,
        forecast_days=forecast_days,
    )

    debug = pipe.predict_with_components(target_rows)
    now = datetime.now(timezone.utc)

    hourly = []
    for ts in debug.y_final.index:
        hourly.append(
            HourlyDebug(
                timestamp=ts.to_pydatetime(),
                y_base=float(debug.y_base.loc[ts]),
                y_spike_hi=float(debug.y_spike_hi.loc[ts]),
                y_spike_lo=float(debug.y_spike_lo.loc[ts]),
                prob_hi=float(debug.prob_hi.loc[ts]),
                prob_lo=float(debug.prob_lo.loc[ts]),
                risk_hi=float(debug.risk_hi.loc[ts]),
                risk_lo=float(debug.risk_lo.loc[ts]),
                y_final=float(debug.y_final.loc[ts]),
            )
        )

    return ForecastDebugResponse(
        target_date=target,
        model_version=pipe.bundle.version,
        forecast_made_at=now,
        n_hours=len(hourly),
        hourly=hourly,
    )


# Старый режим оставляем для research/debug.
def _records_to_df(records: list) -> pd.DataFrame:
    rows = [r.model_dump() for r in records]
    df = pd.DataFrame(rows)

    if "timestamp" not in df.columns:
        raise HTTPException(400, "history records must contain 'timestamp'")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    return df


@router.post("/from-history", response_model=ForecastResponse)
def forecast_from_history(req: ForecastRequest, pipe: PipelineDep) -> ForecastResponse:
    df = _records_to_df(req.history)

    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            422,
            detail={
                "code": "missing_columns",
                "extra": {"missing": missing, "required": list(INPUT_COLUMNS)},
            },
        )

    ctx = FeatureContext(
        as_of=_ensure_utc(req.as_of),
        target_date=pd.Timestamp(req.target_date, tz="UTC"),
    )

    feats = build_feature_frame(df, pipe.bundle.feature_params, ctx)
    target_rows = _slice_target_rows(feats, req.target_date)

    y = pipe.predict(target_rows)
    now = datetime.now(timezone.utc)

    return ForecastResponse(
        target_date=req.target_date,
        model_version=pipe.bundle.version,
        forecast_made_at=now,
        n_hours=len(y),
        hourly=[
            HourlyForecast(timestamp=ts.to_pydatetime(), predicted_price=float(v))
            for ts, v in y.items()
        ],
    )
