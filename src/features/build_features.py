"""build_feature_frame — единая FE функция для training и inference.

Контракт:
    build_feature_frame(history, params, ctx) -> DataFrame

Inference flow:
    history = read_master_frame(end=ctx.as_of + 1d)
    feats = build_feature_frame(history, params, ctx)
    target_rows = feats.loc[ctx.target_date]
    y_pred = model.predict(target_rows)

Training flow:
    history = read_master_frame()
    feats = build_feature_frame(history, params, ctx)
    train_X = feats.loc[:cutoff].dropna()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.features.feature_spec import (
    FEATURE_LIST,
    INPUT_COLUMNS,
    TARGET,
    assert_input_complete,
)
from src.features.holidays_nl import nl_holidays
from src.features.params import FittedFeatureParams


# ──────────────────────────────────────────────────────────────────────────
# Context
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureContext:
    """Аудит-метка для FE-вызова.

    `as_of` — момент, "из которого" делается прогноз. Это ДОКУМЕНТАЦИОННОЕ
    поле (попадает в логи / metadata bundle), а не жёсткая отсечка строк.
    Все transforms в build_feature_frame пользуются ТОЛЬКО `.shift()`, то
    есть фичи в момент T зависят строго от прошлого относительно T —
    никакая будущая строка истории не может leak-нуть в фичи прошлого.
    """

    as_of: pd.Timestamp
    target_date: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        as_of = pd.Timestamp(self.as_of)
        if as_of.tz is None:
            as_of = as_of.tz_localize("UTC")
        else:
            as_of = as_of.tz_convert("UTC")
        object.__setattr__(self, "as_of", as_of)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def build_feature_frame(
    history: pd.DataFrame,
    params: FittedFeatureParams,
    ctx: FeatureContext,
) -> pd.DataFrame:
    """Собирает FEATURE_LIST из master history.

    Parameters
    ----------
    history : DataFrame, hourly DatetimeIndex (UTC), колонки INPUT_COLUMNS
    params  : FittedFeatureParams, fit-нутые на train
    ctx     : FeatureContext с as_of (UTC)

    Returns
    -------
    DataFrame с тем же индексом, что history.loc[:as_of], и FEATURE_LIST колонками.
    Ранние строки могут содержать NaN из-за лагов — это работа caller-а.
    """
    assert_input_complete(history)

    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("history must have a DatetimeIndex")
    if history.index.tz is None:
        history = history.copy()
        history.index = history.index.tz_localize("UTC")

    df = history.copy().sort_index()

    p = df[TARGET]

    # Price lags (same hour history)
    for d in [1, 2, 7, 14, 21, 28]:
        df[f"lag_{d}d"] = p.shift(d * 24)
    df["lag_prev_hour"] = p.shift(24 + 1)
    df["lag_next_hour"] = p.shift(24 - 1)
    df["lag_2d_next"]   = p.shift(48 - 1)

    # Rolling stats (regime detection)
    p1 = p.shift(24)
    for w in [7, 14, 30]:
        df[f"roll_{w}d_mean"] = p1.rolling(w * 24).mean()
        if w in (7, 14):
            df[f"roll_{w}d_std"] = p1.rolling(w * 24).std()
    df["price_vol_3d_lag1d"] = p1.rolling(3 * 24).std()
    df["price_vol_7d_lag1d"] = p1.rolling(7 * 24).std()

    # Commodity & cross-border DA
    df["gas_lag_1d"]   = df["gas_price"].shift(24)
    df["de_da_lag_1d"] = df["de_day_ahead_price"].shift(24)
    df["be_da_lag_1d"] = df["be_day_ahead_price"].shift(24)
    df["fr_da_lag_1d"] = df["fr_day_ahead_price"].shift(24)
    df["de_da_lag_7d"] = df["de_day_ahead_price"].shift(24 * 7)
    df["be_da_lag_7d"] = df["be_day_ahead_price"].shift(24 * 7)
    df["fr_da_lag_7d"] = df["fr_day_ahead_price"].shift(24 * 7)
    df["spread_nl_de_lag1d"] = p.shift(24) - df["de_day_ahead_price"].shift(24)
    df["spread_nl_be_lag1d"] = p.shift(24) - df["be_day_ahead_price"].shift(24)
    df["spread_nl_fr_lag1d"] = p.shift(24) - df["fr_day_ahead_price"].shift(24)

    # Imbalance prices
    df["imb_long_lag_1d"]  = df["imbalance_price_long"].shift(24)
    df["imb_short_lag_1d"] = df["imbalance_price_short"].shift(24)
    df["imb_long_lag_7d"]  = df["imbalance_price_long"].shift(24 * 7)
    df["imb_spread_lag1d"] = df["imb_long_lag_1d"] - df["imb_short_lag_1d"]

    # Weather actuals (lag-only)
    for col in ["temperature_c", "wind_ms", "solar_radiation"]:
        df[f"{col}_lag_1d"] = df[col].shift(24)
    df["temperature_c_lag_7d"] = df["temperature_c"].shift(24 * 7)

    # Generation (forecast + lags)
    for col in ["load_forecast", "wind_forecast_mw"]:
        df[f"{col}_lag_1d"] = df[col].shift(24)
        df[f"{col}_lag_7d"] = df[col].shift(24 * 7)
    df["solar_forecast_mw_lag_1d"] = df["solar_forecast_mw"].shift(24)
    for col in ["net_flow_de_nl", "net_flow_be_nl"]:
        df[f"{col}_lag_1d"] = df[col].shift(24)

    df["residual_load"]        = df["load_forecast"] - df["wind_forecast_mw"] - df["solar_forecast_mw"]
    df["residual_load_lag_1d"] = df["residual_load"].shift(24)
    df["res_ratio"] = (
        df["wind_forecast_mw"]  / params.wind_forecast_max
        + df["solar_forecast_mw"] / params.solar_forecast_max
    ) / 2

    # Calendar features
    df["hour"]       = df.index.hour
    df["dow"]        = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    years = range(int(df.index.min().year), int(df.index.max().year) + 2)
    holidays = nl_holidays(years)
    idx_dates     = df.index.normalize().to_series().dt.date
    next_day_dates = (df.index.normalize() + pd.Timedelta(days=1))
    next_day_dates = pd.Series(next_day_dates, index=df.index).dt.date
    df["is_holiday"]          = idx_dates.isin(holidays).astype(int).values
    df["is_holiday_tomorrow"] = next_day_dates.isin(holidays).astype(int).values

    # Demand drivers
    df["heating_degree"]       = (15 - df["temperature_forecast"]).clip(lower=0)
    df["cooling_degree"]       = (df["temperature_forecast"] - 22).clip(lower=0)
    df["heating_degree_lag1d"] = (15 - df["temperature_c"].shift(24)).clip(lower=0)

    df["darkness"] = (
        df["cloud_cover"].shift(24).fillna(50) / 100
        * (1 - df["solar_radiation_forecast"] / params.solar_radiation_fc_p95)
    )

    df["is_peak_hour"] = df["hour"].isin([15, 16, 17, 18]).astype(int)
    df["is_midday"]    = df["hour"].isin([10, 11, 12, 13, 14]).astype(int)

    df["peak_x_load"]    = df["is_peak_hour"] * df["load_forecast"]
    df["peak_x_heating"] = df["is_peak_hour"] * df["heating_degree"]
    df["peak_x_darkness"] = df["is_peak_hour"] * df["darkness"]

    df["wind_drought"] = (df["wind_forecast_mw"] < params.wind_forecast_p20).astype(int)
    df["peak_x_wind_drought"] = df["is_peak_hour"] * df["wind_drought"]

    df["solar_surplus"] = (df["solar_forecast_mw"] > params.solar_forecast_p80).astype(int)
    df["spike_winter_evening"] = df["is_peak_hour"] * df["heating_degree"] * df["wind_drought"]
    df["spike_neg_weekend"]    = df["is_weekend"] * df["is_midday"] * df["solar_surplus"]

    # Extreme-event features
    df["residual_load_p90_lag1d"] = (
        df["residual_load"].shift(24)
        .rolling(90 * 24, min_periods=30 * 24).quantile(0.90)
    )
    df["residual_load_high"] = (
        df["residual_load"] > df["residual_load_p90_lag1d"]
    ).astype(int)
    df["residual_x_peak"]    = df["residual_load"] * df["is_peak_hour"]
    df["residual_x_heating"] = df["residual_load"] * df["heating_degree"]

    df["low_wind_x_peak"] = df["wind_drought"] * df["is_peak_hour"]
    df["gas_x_peak"]      = df["gas_lag_1d"] * df["is_peak_hour"]
    df["gas_x_heating"]   = df["gas_lag_1d"] * df["heating_degree"]

    df["price_diff_1d_7d"] = df["lag_1d"] - df["lag_7d"]
    df["price_above_30d"]  = df["lag_1d"] - df["roll_30d_mean"]
    df["price_above_7d"]   = df["lag_1d"] - df["roll_7d_mean"]

    df["de_high_lag1d"]   = (df["de_day_ahead_price"].shift(24) > params.de_da_p90).astype(int)
    df["de_high_x_peak"]  = df["de_high_lag1d"] * df["is_peak_hour"]

    df["spread_de_change_1d"] = df["spread_nl_de_lag1d"] - df["spread_nl_de_lag1d"].shift(24)

    return df[FEATURE_LIST].copy()


# ──────────────────────────────────────────────────────────────────────────
# Hash для аудита
# ──────────────────────────────────────────────────────────────────────────

def features_module_hash() -> str:
    """SHA-256 hash содержимого FE модуля.

    При изменении логики FE hash меняется, и старые bundles
    стартуют с явной несовместимостью (бросает исключение).
    """
    import hashlib
    from pathlib import Path

    here = Path(__file__).parent
    files = sorted(here.glob("*.py"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()
