"""Feature engineering — единый контракт между training и inference.

Главное правило: один и тот же `build_feature_frame()` вызывается
из training pipeline и из daily inference. Никакой ad-hoc FE в
notebook'ах в production-флоу.

Глобальные статистики (квантили, max-нормализаторы) fit-ятся ОДИН раз
на train-периоде, сериализуются как `FittedFeatureParams`, и затем
переиспользуются неизменно в inference.

Pure-функция: на одинаковом входе всегда даёт одинаковый выход;
никакого "now()", никакой работы с БД.
"""
from src.features.build_features import (
    FeatureContext,
    build_feature_frame,
)
from src.features.feature_spec import FEATURE_LIST, INPUT_COLUMNS, TARGET
from src.features.params import FittedFeatureParams, fit_feature_params

__all__ = [
    "FeatureContext",
    "build_feature_frame",
    "FEATURE_LIST",
    "INPUT_COLUMNS",
    "TARGET",
    "FittedFeatureParams",
    "fit_feature_params",
]
