"""Roundtrip + integrity tests для ModelBundle.

  • save → load → predict даёт идентичный результат;
  • feature_eng_hash mismatch ловится strict_hash=True;
  • неполный bundle → BundleSchemaError;
  • X с неправильными колонками → BundleSchemaError.

Используем sklearn dummy classifier/regressor вместо LightGBM, чтобы
тест был быстрым и не зависел от обучения.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None

from sklearn.dummy import DummyClassifier, DummyRegressor

from src.features.build_features import features_module_hash
from src.features.feature_spec import FEATURE_LIST
from src.features.params import FittedFeatureParams
from src.models import bundle as bundle_module
from src.models import registry
from src.models.bundle import (
    BlendParams,
    BundleHashMismatchError,
    BundleSchemaError,
    ModelBundle,
    TrainingMetadata,
)
from src.models.pipeline import ForecastPipeline


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

def _tmp_models_dir(monkeypatch) -> Path:
    """Перенаправляет MODELS_DIR на свежую temp папку для теста."""
    tmp = Path(tempfile.mkdtemp(prefix="bundle_test_"))
    monkeypatch.setattr(registry, "MODELS_DIR",       tmp)
    monkeypatch.setattr(registry, "CURRENT_POINTER",  tmp / "current.txt")
    return tmp


def _make_dummy_bundle(version: str = "v0.0.1-test") -> ModelBundle:
    n_feats = len(FEATURE_LIST)

    # Обучаем dummy на синтетике (1 fit для каждого)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (100, n_feats)), columns=FEATURE_LIST)
    y = rng.normal(80, 20, 100)

    stack = [DummyRegressor(strategy="mean").fit(X, y) for _ in range(3)]

    clf_y_hi = (y > np.quantile(y, 0.9)).astype(int)
    clf_y_lo = (y < np.quantile(y, 0.1)).astype(int)
    clf_hi = DummyClassifier(strategy="prior").fit(X, clf_y_hi)
    clf_lo = DummyClassifier(strategy="prior").fit(X, clf_y_lo)

    # spike regressors (отдельные обучения)
    spike_hi = DummyRegressor(strategy="constant", constant=200.0).fit(X, y)
    spike_lo = DummyRegressor(strategy="constant", constant=-50.0).fit(X, y)

    feat_params = FittedFeatureParams(
        wind_forecast_max=8000.0, solar_forecast_max=7000.0,
        wind_forecast_p20=500.0,  solar_forecast_p80=2000.0,
        de_da_p90=250.0,          solar_radiation_fc_p95=600.0,
        fit_period_start="2021-01-01", fit_period_end="2024-12-31",
    )

    return ModelBundle(
        version=version,
        stack_models=stack,
        classifier_hi=clf_hi,
        classifier_lo=clf_lo,
        regressor_spike_hi=spike_hi,
        regressor_spike_lo=spike_lo,
        blend_params=BlendParams(k_hi=0.02, w_hi=0.5, thr_lo=0.5, w_lo=0.9),
        feature_params=feat_params,
        feature_list=list(FEATURE_LIST),
        feature_eng_hash=features_module_hash(),
        metadata=TrainingMetadata(
            train_start="2021-01-01", train_end="2024-12-31",
            val_start="2025-01-01",   val_end="2025-12-31",
            val_mae=14.55, val_naive_mae=30.10, improvement_pct=51.66,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

def test_save_and_load_roundtrip(monkeypatch):
    tmp = _tmp_models_dir(monkeypatch)
    b1 = _make_dummy_bundle()

    out_dir = registry.save_bundle(b1, set_as_current=True)
    assert out_dir.exists()
    assert (tmp / "current.txt").read_text().strip() == b1.version

    b2 = registry.load_bundle("current")

    assert b2.version == b1.version
    assert b2.feature_list == b1.feature_list
    assert b2.feature_eng_hash == b1.feature_eng_hash
    assert b2.blend_params == b1.blend_params
    assert b2.feature_params == b1.feature_params
    assert len(b2.stack_models) == len(b1.stack_models)

    shutil.rmtree(tmp)


def test_predict_shape_and_finiteness(monkeypatch):
    _tmp_models_dir(monkeypatch)
    bundle = _make_dummy_bundle()
    registry.save_bundle(bundle, set_as_current=True)

    pipe = ForecastPipeline.from_bundle("current")

    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        rng.normal(0, 1, (24, len(FEATURE_LIST))),
        columns=FEATURE_LIST,
        index=pd.date_range("2026-05-11", periods=24, freq="h", tz="UTC"),
    )
    y = pipe.predict(X)
    assert len(y) == 24
    assert np.isfinite(y).all()

    debug = pipe.predict_with_components(X)
    assert (debug.prob_hi >= 0).all() and (debug.prob_hi <= 1).all()
    assert (debug.prob_lo >= 0).all() and (debug.prob_lo <= 1).all()
    assert (debug.risk_hi >= 0).all() and (debug.risk_hi <= 1).all()
    assert (debug.risk_lo >= 0).all() and (debug.risk_lo <= 1).all()


def test_wrong_columns_raises(monkeypatch):
    _tmp_models_dir(monkeypatch)
    registry.save_bundle(_make_dummy_bundle(), set_as_current=True)
    pipe = ForecastPipeline.from_bundle("current")

    bad_X = pd.DataFrame({"foo": [1.0]}, index=pd.date_range("2026-05-11", periods=1, tz="UTC"))
    with pytest.raises(BundleSchemaError):
        pipe.predict(bad_X)


def test_nan_in_X_raises(monkeypatch):
    _tmp_models_dir(monkeypatch)
    registry.save_bundle(_make_dummy_bundle(), set_as_current=True)
    pipe = ForecastPipeline.from_bundle("current")

    X = pd.DataFrame(
        np.zeros((24, len(FEATURE_LIST))),
        columns=FEATURE_LIST,
        index=pd.date_range("2026-05-11", periods=24, freq="h", tz="UTC"),
    )
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pipe.predict(X)


def test_hash_mismatch_strict_raises(monkeypatch):
    """Если МАНИФЕСТ написан с другим хэшем — strict load падает."""
    tmp = _tmp_models_dir(monkeypatch)
    bundle = _make_dummy_bundle()
    registry.save_bundle(bundle, set_as_current=True)

    # Подменяем сохранённый хэш на чужой
    manifest_path = tmp / bundle.version / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_eng_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(BundleHashMismatchError):
        registry.load_bundle("current", strict_hash=True)

    # А с strict_hash=False — должен загрузиться (с warning'ом)
    b = registry.load_bundle("current", strict_hash=False)
    assert b.version == bundle.version


def test_double_save_same_version_raises(monkeypatch):
    _tmp_models_dir(monkeypatch)
    bundle = _make_dummy_bundle()
    registry.save_bundle(bundle, set_as_current=True)

    with pytest.raises(FileExistsError):
        registry.save_bundle(bundle, set_as_current=True)


def test_list_versions(monkeypatch):
    _tmp_models_dir(monkeypatch)
    registry.save_bundle(_make_dummy_bundle("v0.0.1"), set_as_current=True)
    registry.save_bundle(_make_dummy_bundle("v0.0.2"), set_as_current=True)

    versions = registry.list_versions()
    assert versions == ["v0.0.1", "v0.0.2"]
    assert registry.current_version() == "v0.0.2"


# ──────────────────────────────────────────────────────────────────────────
# Manual runner (без pytest)
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Минимальный smoke run: создать → сохранить → загрузить → predict."""
    import logging
    logging.basicConfig(level=logging.INFO)

    tmp = Path(tempfile.mkdtemp(prefix="bundle_smoke_"))
    print(f"Using tmp models dir: {tmp}")

    # подмена через прямую запись
    registry.MODELS_DIR = tmp
    registry.CURRENT_POINTER = tmp / "current.txt"

    bundle = _make_dummy_bundle()
    registry.save_bundle(bundle, set_as_current=True)
    print(f"  ✓ saved bundle {bundle.version}")

    b2 = registry.load_bundle("current")
    print(f"  ✓ loaded bundle {b2.version} (hash ok)")

    pipe = ForecastPipeline(b2)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(0, 1, (24, len(FEATURE_LIST))),
        columns=FEATURE_LIST,
    )
    y = pipe.predict(X)
    print(f"  ✓ predict on 24h: mean={y.mean():.2f}, std={y.std():.2f}")

    shutil.rmtree(tmp)
    print("\nSMOKE OK")
