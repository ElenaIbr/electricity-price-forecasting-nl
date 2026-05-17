"""Model artifacts: bundle, registry, forecast pipeline.

Bundle = атомарный артефакт обученной системы. Всё или ничего:
  • 10 stacking LGBM регрессоров,
  • HIGH/LOW spike classifier-ы,
  • HIGH/LOW spike regressor-ы,
  • blend params,
  • fitted feature params (квантили),
  • feature list (порядок ВАЖЕН),
  • feature_eng_hash (для validation),
  • metadata (train period, val MAE, git sha, etc.).

Save: на диск как папка models/{version}/ + плоский набор файлов.
Load: load_bundle('current') читает указатель models/current.txt и
поднимает всё в одно `ModelBundle`.
"""
from src.models.bundle import (
    BlendParams,
    BundleHashMismatchError,
    ModelBundle,
    TrainingMetadata,
)
from src.models.pipeline import ForecastPipeline
from src.models.registry import (
    MODELS_DIR,
    list_versions,
    load_bundle,
    save_bundle,
    update_current,
)

__all__ = [
    "BlendParams",
    "BundleHashMismatchError",
    "ForecastPipeline",
    "MODELS_DIR",
    "ModelBundle",
    "TrainingMetadata",
    "list_versions",
    "load_bundle",
    "save_bundle",
    "update_current",
]
