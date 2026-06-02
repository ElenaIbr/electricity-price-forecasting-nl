"""No-future-leakage invariant.

Главный тест-контракт FE модуля. Если он зелёный — фичи на любом
timestamp T зависят ТОЛЬКО от данных history.loc[:T] (плюс forecast-колонок,
которые в master frame по построению future-friendly).

Стратегия:
  • генерируем синтетический master history;
  • строим фичи дважды: раз на отрезанной (через as_of) истории,
    раз на полной — но с тем же as_of;
  • обе версии должны совпадать до байта на пересечении индексов.

Дополнительно проверяем:
  • output ровно FEATURE_LIST в правильном порядке;
  • количество строк = количество строк history.loc[:as_of];
  • заведомые leakage-сценарии (когда мы случайно используем будущее)
    ловятся через canary: добавляем в "будущую" часть outliers
    и убеждаемся, что они не просочились в фичи прошлого.

Запуск:
    pytest src/features/tests/test_no_leakage.py -v
или:
    python -m src.features.tests.test_no_leakage
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import FeatureContext, build_feature_frame
from src.features.feature_spec import FEATURE_LIST, INPUT_COLUMNS, TARGET
from src.features.params import fit_feature_params


# ──────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ──────────────────────────────────────────────────────────────────────────

def _synthetic_history(
    start: str = "2023-01-01",
    end: str = "2024-12-31 23:00",
    seed: int = 42,
) -> pd.DataFrame:
    """Достаточно длинная (≥365d) hourly история с правдоподобной структурой
    данных, чтобы все 84 фичи могли быть посчитаны (включая 30d rolling)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(idx)

    hour_of_day = idx.hour.values
    day_of_year = idx.dayofyear.values

    base_price = 80 + 30 * np.sin(2 * np.pi * hour_of_day / 24)
    seasonal = 20 * np.sin(2 * np.pi * day_of_year / 365)
    noise = rng.normal(0, 15, n)
    nl_price = base_price + seasonal + noise

    data = {
        TARGET: nl_price,
        "be_day_ahead_price": nl_price + rng.normal(0, 5, n),
        "de_day_ahead_price": nl_price + rng.normal(0, 8, n),
        "fr_day_ahead_price": nl_price + rng.normal(0, 6, n),
        "gas_price":          30 + rng.normal(0, 3, n),
        "net_flow_de_nl":     rng.normal(500, 1000, n),
        "net_flow_be_nl":     rng.normal(0, 800, n),
        "imbalance_price_long":  nl_price + rng.normal(0, 30, n),
        "imbalance_price_short": nl_price - rng.normal(0, 30, n),
        "load_forecast":      11000 + 3000 * np.sin(2 * np.pi * hour_of_day / 24) + rng.normal(0, 500, n),
        "wind_forecast_mw":   np.clip(2000 + rng.normal(0, 1500, n), 0, 8000),
        "solar_forecast_mw":  np.clip(2500 * np.sin(np.pi * hour_of_day / 24), 0, None) + rng.normal(0, 100, n).clip(min=0),
        "temperature_c":      10 + 8 * np.sin(2 * np.pi * day_of_year / 365) + rng.normal(0, 3, n),
        "wind_ms":            5 + rng.normal(0, 2, n).clip(min=0),
        "solar_radiation":    np.clip(800 * np.sin(np.pi * hour_of_day / 24), 0, None),
        "cloud_cover":        rng.uniform(0, 100, n),
        "humidity":           rng.uniform(40, 95, n),
        "temperature_forecast":      10 + 8 * np.sin(2 * np.pi * day_of_year / 365) + rng.normal(0, 3, n),
        "wind_speed_forecast":       5 + rng.normal(0, 2, n).clip(min=0),
        "solar_radiation_forecast":  np.clip(800 * np.sin(np.pi * hour_of_day / 24), 0, None),
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    # sanity-check: все INPUT_COLUMNS в наличии
    assert all(c in df.columns for c in INPUT_COLUMNS), (
        f"Missing in synthetic: {set(INPUT_COLUMNS) - set(df.columns)}"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

def test_output_shape_and_columns():
    history = _synthetic_history()
    params = fit_feature_params(history)
    ctx = FeatureContext(as_of=history.index[-1])

    feats = build_feature_frame(history, params, ctx)

    assert list(feats.columns) == FEATURE_LIST, "Columns must match FEATURE_LIST exactly"
    assert len(feats) == len(history), "Output rows must match input rows ≤ as_of"


def test_no_future_leakage_basic():
    """Главный invariant: features at T не зависят от строк > T в history.

    build_feature_frame НЕ режет history по as_of — но фичи прошлого должны
    быть идентичны при любом продолжении (или его отсутствии).
    """
    history = _synthetic_history()
    params = fit_feature_params(history.loc[:"2024-06-30"])

    cutoff = pd.Timestamp("2024-06-30 23:00", tz="UTC")
    ctx = FeatureContext(as_of=cutoff)

    short = history.loc[:cutoff]
    full = history

    feats_short = build_feature_frame(short, params, ctx)
    feats_full  = build_feature_frame(full,  params, ctx)

    # На срезе [:cutoff] обе версии должны совпасть до байта.
    pd.testing.assert_frame_equal(
        feats_short.loc[:cutoff],
        feats_full.loc[:cutoff],
        check_exact=False, atol=1e-10, rtol=1e-10,
    )


def test_no_leakage_via_corrupted_future():
    """Canary: подменяем будущую часть истории на абсурдные значения.
    Фичи прошлого (на индексе ≤ cutoff) должны остаться побайтово прежними.
    """
    history = _synthetic_history()
    params = fit_feature_params(history.loc[:"2024-06-30"])

    cutoff = pd.Timestamp("2024-06-30 23:00", tz="UTC")
    ctx = FeatureContext(as_of=cutoff)

    feats_clean = build_feature_frame(history, params, ctx)

    corrupted = history.copy()
    future_mask = corrupted.index > cutoff
    corrupted.loc[future_mask, INPUT_COLUMNS] = 999999.0

    feats_corrupted = build_feature_frame(corrupted, params, ctx)

    pd.testing.assert_frame_equal(
        feats_clean.loc[:cutoff],
        feats_corrupted.loc[:cutoff],
        check_exact=False, atol=1e-10, rtol=1e-10,
    )


def test_dropna_yields_dense_block():
    """После warmup (28d для lag_28d, 90d для residual_p90) фичи должны быть
    плотными — никаких дырок NaN внутри."""
    history = _synthetic_history()
    params = fit_feature_params(history)
    ctx = FeatureContext(as_of=history.index[-1])

    feats = build_feature_frame(history, params, ctx)
    # warmup ~120 days
    feats_warm = feats.loc["2023-06-01":]
    assert not feats_warm.isna().any().any(), (
        f"NaN после warmup в фичах: "
        f"{feats_warm.columns[feats_warm.isna().any()].tolist()}"
    )


def test_params_are_deterministic_on_same_train():
    """Fit-нутые параметры должны быть детерминированы на одном train-окне."""
    history = _synthetic_history()
    p1 = fit_feature_params(history.loc[:"2024-06-30"])
    p2 = fit_feature_params(history.loc[:"2024-06-30"])
    assert p1 == p2


def test_output_index_matches_input():
    """Output index = input index. Caller сам режет по target_date."""
    history = _synthetic_history()
    params = fit_feature_params(history.loc[:"2024-06-30"])
    ctx = FeatureContext(as_of=pd.Timestamp("2024-06-30 23:00", tz="UTC"))

    feats = build_feature_frame(history, params, ctx)
    pd.testing.assert_index_equal(feats.index, history.index)


# ──────────────────────────────────────────────────────────────────────────
# Manual runner
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_output_shape_and_columns,
        test_no_future_leakage_basic,
        test_no_leakage_via_corrupted_future,
        test_dropna_yields_dense_block,
        test_params_are_deterministic_on_same_train,
        test_output_index_matches_input,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as exc:
            print(f"  ✗ {t.__name__}: {exc}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(0 if failures == 0 else 1)
