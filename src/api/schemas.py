"""Dependency helpers for accessing the shared ForecastPipeline.

The pipeline is loaded once during app startup and reused across requests
for better performance.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.features.feature_spec import INPUT_COLUMNS


# ──────────────────────────────────────────────────────────────────────────
# Health / Info
# ──────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    api_version: str
    model_version: str | None = None


class BlendParamsOut(BaseModel):
    k_hi:   float
    w_hi:   float
    thr_lo: float
    w_lo:   float


class FittedFeatureParamsOut(BaseModel):
    wind_forecast_max:      float
    solar_forecast_max:     float
    wind_forecast_p20:      float
    solar_forecast_p80:     float
    de_da_p90:              float
    solar_radiation_fc_p95: float
    fit_period_start:       str
    fit_period_end:         str


class TrainingMetadataOut(BaseModel):
    architecture:    str
    train_start:     str
    train_end:       str
    val_start:       str
    val_end:         str
    val_mae:         float
    val_naive_mae:   float
    improvement_pct: float
    git_sha:         str = ""
    fit_timestamp:   str = ""


class InfoResponse(BaseModel):
    model_version:    str
    feature_eng_hash: str
    n_stack_models:   int
    n_features:       int
    blend_params:     BlendParamsOut
    feature_params:   FittedFeatureParamsOut
    metadata:         TrainingMetadataOut


class FeaturesInfoResponse(BaseModel):
    """Какие колонки нужны на вход и какие фичи получает модель."""
    required_input_columns: list[str]
    feature_list:           list[str]


# ──────────────────────────────────────────────────────────────────────────
# Forecast
# ──────────────────────────────────────────────────────────────────────────

class HistoryRecord(BaseModel):
    """Одна строка master frame.

    extra='allow', чтобы клиент мог слать дополнительные колонки —
    они просто будут проигнорированы FE (build_feature_frame смотрит
    только в INPUT_COLUMNS).
    """
    model_config = ConfigDict(extra="allow")

    timestamp: datetime


class ForecastRequest(BaseModel):
    """Запрос на forecast D+1.

    history     — hourly master records, минимум lookback 35 дней.
    as_of       — момент запуска прогноза (UTC). Всё после этого отсекается.
    target_date — день для прогноза (D+1 относительно as_of).
    """
    history:     list[HistoryRecord] = Field(..., min_length=24)
    as_of:       datetime
    target_date: date


class HourlyForecast(BaseModel):
    timestamp:        datetime
    predicted_price:  float


class ForecastResponse(BaseModel):
    target_date:      date
    model_version:    str
    forecast_made_at: datetime
    n_hours:          int
    hourly:           list[HourlyForecast]


# ──────────────────────────────────────────────────────────────────────────
# Debug
# ──────────────────────────────────────────────────────────────────────────

class HourlyDebug(BaseModel):
    timestamp:    datetime
    y_base:       float
    y_spike_hi:   float
    y_spike_lo:   float
    prob_hi:      float
    prob_lo:      float
    risk_hi:      float
    risk_lo:      float
    y_final:      float


class ForecastDebugResponse(BaseModel):
    target_date:      date
    model_version:    str
    forecast_made_at: datetime
    n_hours:          int
    hourly:           list[HourlyDebug]


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    code:   str = "internal_error"
    extra:  dict[str, Any] | None = None
