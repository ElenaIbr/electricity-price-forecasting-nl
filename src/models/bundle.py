"""ModelBundle — atomic dataclass для production-артефактов.

Bundle сериализуется как папка models/{version}/ с плоским набором файлов.
Никаких pickle-моноблоков: каждый компонент в своём файле, чтобы было
можно diff-ать версии (например, метаданные).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.features.params import FittedFeatureParams


# ──────────────────────────────────────────────────────────────────────────
# Блоки внутри bundle
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlendParams:
    """Параметры asymmetric spike blend (выход joint optimization)."""

    k_hi:   float   # top-k% rows для HIGH risk ranking, e.g. 0.02
    w_hi:   float   # вес HIGH-коррекции, e.g. 0.5
    thr_lo: float   # порог prob_lo для LOW risk, e.g. 0.567
    w_lo:   float   # вес LOW-коррекции, e.g. 0.9


@dataclass
class TrainingMetadata:
    """Метаданные тренировки. Сериализуется в metadata.json."""

    architecture:    str = "AveragingEnsemble10LGBM + AsymmetricSpikeBlend"
    train_start:     str = ""
    train_end:       str = ""
    val_start:       str = ""
    val_end:         str = ""
    val_mae:         float = float("nan")
    val_rmse:        float = float("nan")
    val_smape:       float = float("nan")
    val_naive_mae:   float = float("nan")
    improvement_pct: float = float("nan")
    git_sha:         str = ""
    python_version:  str = ""
    fit_timestamp:   str = ""
    stack_configs:   list[dict] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────

class BundleHashMismatchError(RuntimeError):
    """feature_eng_hash в bundle не совпадает с текущим src/features/."""


class BundleSchemaError(RuntimeError):
    """Bundle не имеет ожидаемых файлов / колонок."""


# ──────────────────────────────────────────────────────────────────────────
# Главный artifact
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ModelBundle:
    """Атомарный production-артефакт.

    Все компоненты должны соответствовать друг другу:
      • stack_models обучены на FEATURE_LIST (в этом порядке);
      • classifier_hi/lo обучены на тех же фичах;
      • regressor_spike_hi/lo тоже;
      • blend_params получены joint search'ем на val;
      • feature_params (FittedFeatureParams) использовались при FE на тех
        самых данных, на которых обучались модели;
      • feature_eng_hash зафиксирован — несовместимость FE-кода
        отлавливается при load_bundle().
    """

    version: str

    # Регрессоры основного ансамбля (10 шт.)
    stack_models: list[Any]

    # Spike компоненты
    classifier_hi:      Any   # LGBMClassifier
    classifier_lo:      Any
    regressor_spike_hi: Any   # LGBMRegressor
    regressor_spike_lo: Any

    # Параметры blend
    blend_params: BlendParams

    # Параметры FE (квантили)
    feature_params: FittedFeatureParams

    # Список фич (порядок ИНВАРИАНТЕН)
    feature_list: list[str]

    # Hash содержимого src/features/ — гарантия совместимости
    feature_eng_hash: str

    # Метаданные
    metadata: TrainingMetadata
