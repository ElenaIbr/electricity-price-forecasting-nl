"""ForecastPipeline — composition stack ensemble + spike blend.

Контракт inference:
    pipe = ForecastPipeline.from_bundle("current")
    y_pred = pipe.predict(X)            # 24 hourly предикта на D+1

X должен содержать колонки в точности bundle.feature_list, в правильном
порядке. Этим занимается build_feature_frame в src/features/.

Дополнительно есть `predict_with_components(X)` — возвращает все
промежуточные сигналы (y_base, prob_hi, prob_lo, ...), удобно для
мониторинга и debugging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.bundle import BundleSchemaError, ModelBundle
from src.models.registry import load_bundle
from src.models.spike_blend import asymmetric_blend, make_prob_risk, make_topk_risk

logger = logging.getLogger(__name__)


@dataclass
class PredictionDebug:
    """Все промежуточные сигналы. Удобно для дашбордов / алертов."""

    y_base:      pd.Series
    y_spike_hi:  pd.Series
    y_spike_lo:  pd.Series
    prob_hi:     pd.Series
    prob_lo:     pd.Series
    risk_hi:     pd.Series
    risk_lo:     pd.Series
    y_final:     pd.Series


class ForecastPipeline:
    """Inference композиция: stack ensemble → classifiers → asymmetric blend."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    # ────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────

    @classmethod
    def from_bundle(cls, version: str = "current", strict_hash: bool = True) -> "ForecastPipeline":
        return cls(load_bundle(version, strict_hash=strict_hash))

    # ────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────

    def _validate_X(self, X: pd.DataFrame) -> pd.DataFrame:
        expected = self.bundle.feature_list
        if list(X.columns) != expected:
            missing = set(expected) - set(X.columns)
            extra   = set(X.columns) - set(expected)
            raise BundleSchemaError(
                f"X columns do not match bundle.feature_list:\n"
                f"  missing: {sorted(missing)}\n"
                f"  extra:   {sorted(extra)}\n"
                f"  order match: {list(X.columns) == expected}"
            )
        if X.isna().any().any():
            n_nan = int(X.isna().sum().sum())
            cols  = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"X contains {n_nan} NaN values across columns {cols[:5]}..."
            )
        return X

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Возвращает финальный y_final (после blend)."""
        return self.predict_with_components(X).y_final

    def predict_with_components(self, X: pd.DataFrame) -> PredictionDebug:
        b = self.bundle
        X = self._validate_X(X)

        # 1. Stack ensemble — простое усреднение 10 LGBM
        stack_preds = np.column_stack([m.predict(X) for m in b.stack_models])
        y_base = pd.Series(stack_preds.mean(axis=1), index=X.index, name="y_base")

        # 2. Classifier probabilities (P(spike))
        prob_hi = pd.Series(
            b.classifier_hi.predict_proba(X)[:, 1], index=X.index, name="prob_hi"
        )
        prob_lo = pd.Series(
            b.classifier_lo.predict_proba(X)[:, 1], index=X.index, name="prob_lo"
        )

        # 3. Spike-specific regressors
        y_spike_hi = pd.Series(
            b.regressor_spike_hi.predict(X), index=X.index, name="y_spike_hi"
        )
        y_spike_lo = pd.Series(
            b.regressor_spike_lo.predict(X), index=X.index, name="y_spike_lo"
        )

        # 4. Asymmetric blend
        bp = b.blend_params
        risk_hi = make_topk_risk(prob_hi, bp.k_hi)
        risk_lo = make_prob_risk(prob_lo, bp.thr_lo)
        y_final = asymmetric_blend(
            y_base, y_spike_hi, y_spike_lo,
            prob_hi, prob_lo,
            bp.k_hi, bp.w_hi, bp.thr_lo, bp.w_lo,
        )
        y_final.name = "y_final"

        return PredictionDebug(
            y_base=y_base,
            y_spike_hi=y_spike_hi,
            y_spike_lo=y_spike_lo,
            prob_hi=prob_hi,
            prob_lo=prob_lo,
            risk_hi=risk_hi,
            risk_lo=risk_lo,
            y_final=y_final,
        )
