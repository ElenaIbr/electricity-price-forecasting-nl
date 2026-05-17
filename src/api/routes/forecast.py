"""POST /forecast — основной inference endpoint.

Контракт:
    request:  history (list of hourly records) + as_of + target_date
    response: 24 hourly прогнозов на target_date

Шаги внутри:
  1. Парсим history → DataFrame с DatetimeIndex (UTC).
  2. Проверяем наличие INPUT_COLUMNS.
  3. build_feature_frame(history, params, ctx=as_of)
  4. Вырезаем 24 строки, соответствующие target_date.
  5. pipeline.predict(X)
  6. Сериализуем → ForecastResponse

Никакой персистенции в БД на этом этапе — это задача
inference pipeline (`src/inference/daily.py`).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.dependencies import PipelineDep
from src.api.schemas import (
    ForecastDebugResponse,
    ForecastRequest,
    ForecastResponse,
    HourlyDebug,
    HourlyForecast,
)
from src.features.build_features import FeatureContext, build_feature_frame
from src.features.feature_spec import INPUT_COLUMNS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["forecast"], prefix="/forecast")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _records_to_df(records: list) -> pd.DataFrame:
    """Pydantic models → DataFrame с UTC DatetimeIndex."""
    rows = [r.model_dump() for r in records]
    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns:
        raise HTTPException(400, "history records must contain 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _ensure_utc(ts: datetime) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _slice_target_day(feats: pd.DataFrame, target_date) -> pd.DataFrame:
    target_str = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    sub = feats.loc[target_str:target_str]
    if len(sub) == 0:
        raise HTTPException(
            400,
            f"No feature rows for target_date={target_str}. "
            f"Provided history may not extend to target day. "
            f"Range: {feats.index.min()} … {feats.index.max()}",
        )
    return sub


def _build_feature_frame_for_request(
    req: ForecastRequest,
    pipe,
):
    df = _records_to_df(req.history)

    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            422,
            detail={
                "code":   "missing_columns",
                "extra":  {"missing": missing, "required": list(INPUT_COLUMNS)},
                "detail": (
                    f"history is missing {len(missing)} required input columns "
                    f"(see /info/features for full list)"
                ),
            },
        )

    ctx = FeatureContext(
        as_of=_ensure_utc(req.as_of),
        target_date=pd.Timestamp(req.target_date, tz="UTC"),
    )

    try:
        feats = build_feature_frame(df, pipe.bundle.feature_params, ctx)
    except Exception as exc:
        logger.exception("FE failed: %s", exc)
        raise HTTPException(500, detail=f"Feature engineering failed: {exc}")

    target_rows = _slice_target_day(feats, req.target_date)

    if target_rows.isna().any().any():
        n_nan = int(target_rows.isna().sum().sum())
        cols = target_rows.columns[target_rows.isna().any()].tolist()
        raise HTTPException(
            422,
            detail={
                "code": "nan_in_features",
                "extra": {"n_nan": n_nan, "cols_with_nan": cols},
                "detail": (
                    "Computed features for target day contain NaN. "
                    "Likely the provided history is too short (need ≥ 35 days "
                    "for warmup of lag/rolling features)."
                ),
            },
        )

    return target_rows


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────

@router.post("", response_model=ForecastResponse)
def forecast(req: ForecastRequest, pipe: PipelineDep) -> ForecastResponse:
    target_rows = _build_feature_frame_for_request(req, pipe)
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


@router.post("/debug", response_model=ForecastDebugResponse)
def forecast_debug(req: ForecastRequest, pipe: PipelineDep) -> ForecastDebugResponse:
    """Тот же forecast, но с разложением на компоненты.

    Полезно для:
      • визуализации в дашбордах,
      • алертов на резкие сдвиги между y_base и y_final,
      • debug несовпадения между обучением и inference.
    """
    target_rows = _build_feature_frame_for_request(req, pipe)
    debug = pipe.predict_with_components(target_rows)
    now = datetime.now(timezone.utc)

    hourly = []
    for ts in debug.y_final.index:
        hourly.append(HourlyDebug(
            timestamp   = ts.to_pydatetime(),
            y_base      = float(debug.y_base.loc[ts]),
            y_spike_hi  = float(debug.y_spike_hi.loc[ts]),
            y_spike_lo  = float(debug.y_spike_lo.loc[ts]),
            prob_hi     = float(debug.prob_hi.loc[ts]),
            prob_lo     = float(debug.prob_lo.loc[ts]),
            risk_hi     = float(debug.risk_hi.loc[ts]),
            risk_lo     = float(debug.risk_lo.loc[ts]),
            y_final     = float(debug.y_final.loc[ts]),
        ))

    return ForecastDebugResponse(
        target_date=req.target_date,
        model_version=pipe.bundle.version,
        forecast_made_at=now,
        n_hours=len(hourly),
        hourly=hourly,
    )
