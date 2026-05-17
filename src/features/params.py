"""Fitted feature parameters.

Глобальные статистики, которые при наивном использовании создают
train/serve skew. Здесь они fit-ятся ОДИН раз на train-окне и
далее переиспользуются неизменно — и в training, и в inference.

Сериализация: JSON, чтобы было видно глазами при ревью bundle'а.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FittedFeatureParams:
    """Глобальные стат-параметры, нужные для FE.

    Все значения — float, чтобы сериализация была тривиальной.
    """

    wind_forecast_max:      float
    solar_forecast_max:     float   # max после замены 0 на NaN (как в notebook)
    wind_forecast_p20:      float
    solar_forecast_p80:     float
    de_da_p90:              float
    solar_radiation_fc_p95: float

    fit_period_start: str            # ISO date, для аудита
    fit_period_end:   str

    # ────────────────────────────────────────
    # IO
    # ────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FittedFeatureParams":
        data = json.loads(Path(path).read_text())
        return cls(**data)


def fit_feature_params(
    train_history: pd.DataFrame,
    fit_period_start: str | None = None,
    fit_period_end: str | None = None,
) -> FittedFeatureParams:
    """Fit-ит глобальные параметры FE на train-окне.

    Ожидает DataFrame с колонками из INPUT_COLUMNS, индекс — DatetimeIndex.

    `fit_period_start/end` — опциональные ISO-даты для записи в metadata
    (по факту используется весь переданный train_history).
    """
    if train_history.empty:
        raise ValueError("Cannot fit on empty history")

    wind = train_history["wind_forecast_mw"]
    solar = train_history["solar_forecast_mw"]
    de_da = train_history["de_day_ahead_price"]
    solar_rad_fc = train_history["solar_radiation_forecast"]

    return FittedFeatureParams(
        wind_forecast_max      = float(wind.max()),
        # NB: notebook делает .replace(0, NaN).max() — у солнечной есть
        # ночные нули, и max после такой замены = реальный пиковый ампл.
        solar_forecast_max     = float(solar.replace(0, pd.NA).dropna().max()),
        wind_forecast_p20      = float(wind.quantile(0.20)),
        solar_forecast_p80     = float(solar.quantile(0.80)),
        de_da_p90              = float(de_da.quantile(0.90)),
        solar_radiation_fc_p95 = float(max(solar_rad_fc.quantile(0.95), 1.0)),
        fit_period_start = fit_period_start or str(train_history.index.min().date()),
        fit_period_end   = fit_period_end   or str(train_history.index.max().date()),
    )
