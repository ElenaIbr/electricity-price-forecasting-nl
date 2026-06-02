"""Канонический список входных колонок и финальных фич.

При изменении этого файла НЕОБХОДИМО пересобрать и переучить модель —
старые bundles будут невалидны (это проверяется через feature_eng_hash).
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Target
# ──────────────────────────────────────────────────────────────────────────

TARGET = "nl_day_ahead_price"


# ──────────────────────────────────────────────────────────────────────────
# Required columns в input master frame
# (сейчас собирается из master_hourly_*.csv в EDA pipeline; в будущем —
# из curated.master_hourly view в Postgres)
# ──────────────────────────────────────────────────────────────────────────

INPUT_COLUMNS: list[str] = [
    # Targets / cross-border DA prices
    "nl_day_ahead_price",
    "be_day_ahead_price",
    "de_day_ahead_price",
    "fr_day_ahead_price",
    # Commodity
    "gas_price",
    # Cross-border flows
    "net_flow_de_nl",
    "net_flow_be_nl",
    # Imbalance prices
    "imbalance_price_long",
    "imbalance_price_short",
    # Demand
    "load_forecast",
    # Generation forecasts (D+1 friendly)
    "wind_forecast_mw",
    "solar_forecast_mw",
    # Weather actuals (lag-only)
    "temperature_c",
    "wind_ms",
    "solar_radiation",
    "cloud_cover",
    "humidity",
    # Weather forecasts (D+1 friendly)
    "temperature_forecast",
    "wind_speed_forecast",
    "solar_radiation_forecast",
]


# ──────────────────────────────────────────────────────────────────────────
# Финальный список фич, в порядке, ожидаемом моделью
# ──────────────────────────────────────────────────────────────────────────

FEATURE_LIST: list[str] = [
    # Price lags
    "lag_1d", "lag_2d", "lag_7d", "lag_14d", "lag_21d", "lag_28d",
    "lag_prev_hour", "lag_next_hour", "lag_2d_next",
    # Rolling и volatility
    "roll_7d_mean", "roll_7d_std", "roll_14d_mean", "roll_14d_std", "roll_30d_mean",
    "price_vol_3d_lag1d", "price_vol_7d_lag1d",
    # Commodity и cross-border DA
    "gas_lag_1d",
    "de_da_lag_1d", "be_da_lag_1d", "fr_da_lag_1d",
    "de_da_lag_7d", "be_da_lag_7d", "fr_da_lag_7d",
    "spread_nl_de_lag1d", "spread_nl_be_lag1d", "spread_nl_fr_lag1d",
    # Imbalance
    "imb_long_lag_1d", "imb_short_lag_1d", "imb_long_lag_7d", "imb_spread_lag1d",
    # Weather actuals (lag-only)
    "temperature_c_lag_1d", "temperature_c_lag_7d",
    "wind_ms_lag_1d", "solar_radiation_lag_1d",
    # Weather forecasts (D+1 friendly)
    "temperature_forecast", "wind_speed_forecast", "solar_radiation_forecast",
    # Generation
    "load_forecast", "load_forecast_lag_1d", "load_forecast_lag_7d",
    "wind_forecast_mw", "wind_forecast_mw_lag_1d", "wind_forecast_mw_lag_7d",
    "solar_forecast_mw", "solar_forecast_mw_lag_1d",
    "res_ratio", "residual_load_lag_1d",
    # Cross-border flows (lag)
    "net_flow_de_nl_lag_1d", "net_flow_be_nl_lag_1d",
    # Calendar
    "hour", "dow", "month", "is_weekend",
    "is_holiday", "is_holiday_tomorrow",
    # Demand drivers
    "heating_degree", "cooling_degree", "heating_degree_lag1d", "darkness",
    "is_peak_hour", "is_midday",
    "peak_x_load", "peak_x_heating", "peak_x_darkness",
    "wind_drought", "peak_x_wind_drought",
    "spike_winter_evening", "spike_neg_weekend", "solar_surplus",
    # Scarcity / extreme-event features
    "residual_load_p90_lag1d", "residual_load_high",
    "residual_x_peak", "residual_x_heating",
    "low_wind_x_peak",
    "gas_x_peak", "gas_x_heating",
    "price_diff_1d_7d", "price_above_30d", "price_above_7d",
    "de_high_lag1d", "de_high_x_peak",
    "spread_de_change_1d",
]


N_FEATURES: int = len(FEATURE_LIST)


def assert_input_complete(df) -> None:
    """Проверка, что master frame имеет все нужные колонки."""
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
