"""Asymmetric spike blend helpers.

Чистые pure-функции, используются в ForecastPipeline.predict.

Дизайн (как в notebook):
  • HIGH spike — редкий и плохо предсказуемый. Используем top-k%
    ранжирование вероятностей: только самые высокие prob_hi подталкивают
    предикт к спайк-регрессору. Остальные не трогаем.
  • LOW spike — более стабильный (повторяемые условия: солнце × выходные).
    Используем threshold-based blend с пропорциональной риском силой.
"""
from __future__ import annotations

import pandas as pd


def make_topk_risk(prob: pd.Series, k_pct: float) -> pd.Series:
    """Возвращает risk-вес ∈ [0, 1] для top k_pct% строк по prob, остальные 0.

    Внутри top-k risk = prob / max_prob (нормировано), что сохраняет
    относительную уверенность classifier-а.
    """
    n_top = max(1, int(len(prob) * k_pct))
    top_idx = prob.sort_values(ascending=False).head(n_top).index
    risk = pd.Series(0.0, index=prob.index)
    if len(top_idx) > 0:
        max_prob = prob.loc[top_idx].max()
        if max_prob > 1e-9:
            risk.loc[top_idx] = prob.loc[top_idx] / max_prob
    return risk


def make_prob_risk(prob: pd.Series, threshold: float) -> pd.Series:
    """Линейный risk-вес: 0 ниже threshold, нормированно растёт до 1 при prob=1."""
    denom = max(1.0 - threshold, 1e-9)
    return ((prob - threshold) / denom).clip(0.0, 1.0)


def asymmetric_blend(
    y_base:      pd.Series,
    y_spike_hi:  pd.Series,
    y_spike_lo:  pd.Series,
    prob_hi:     pd.Series,
    prob_lo:     pd.Series,
    k_hi:        float,
    w_hi:        float,
    thr_lo:      float,
    w_lo:        float,
) -> pd.Series:
    """Финальная формула:

        ŷ = ŷ_base
            + w_hi · risk_hi · (ŷ_spike_hi − ŷ_base)
            + w_lo · risk_lo · (ŷ_spike_lo − ŷ_base)

    risk_hi — top-k% от prob_hi
    risk_lo — threshold-based от prob_lo
    """
    risk_hi = make_topk_risk(prob_hi, k_hi)
    risk_lo = make_prob_risk(prob_lo, thr_lo)
    return (
        y_base
        + w_hi * risk_hi * (y_spike_hi - y_base)
        + w_lo * risk_lo * (y_spike_lo - y_base)
    )
