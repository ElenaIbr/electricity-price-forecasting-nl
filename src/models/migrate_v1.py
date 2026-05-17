"""One-shot migration: legacy notebook bundle (v1.0.0) → новый формат.

Legacy layout (что было сохранено из notebook'а):
    models/v1.0.0/
        averaging_ensemble_10lgbm.joblib       # list[LGBMRegressor]
        spike_classifier_high.joblib
        spike_classifier_low.joblib
        spike_regressor_high.joblib
        spike_regressor_low.joblib
        feature_list.txt
        metadata.json                          # содержит quantiles, blend, configs

Новый layout (см. registry.py):
    models/v1.0.0_migrated/
        stacking/model_*.joblib
        classifier_hi.joblib
        classifier_lo.joblib
        regressor_spike_hi.joblib
        regressor_spike_lo.joblib
        blend_params.json
        feature_params.json
        feature_list.json
        metadata.json
        MANIFEST.json

Запуск:
    python -m src.models.migrate_v1
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from src.features.build_features import features_module_hash
from src.features.feature_spec import FEATURE_LIST as CANONICAL_FEATURE_LIST
from src.features.params import FittedFeatureParams
from src.models.bundle import BlendParams, ModelBundle, TrainingMetadata
from src.models.registry import MODELS_DIR, save_bundle

logger = logging.getLogger(__name__)


LEGACY_VERSION = "v1.0.0"
NEW_VERSION    = "v1.0.0-migrated"


def migrate() -> Path:
    legacy_dir = MODELS_DIR / LEGACY_VERSION
    if not legacy_dir.exists():
        raise FileNotFoundError(f"Legacy bundle not found: {legacy_dir}")

    # ────────── Парсим legacy metadata.json ──────────
    legacy_meta = json.loads((legacy_dir / "metadata.json").read_text())

    # Feature list: проверяем что совпадает с каноническим FEATURE_LIST
    legacy_feats = legacy_meta.get("feat_full") or legacy_meta.get("feature_list")
    if legacy_feats is None:
        raise ValueError("metadata.json has no feat_full/feature_list")
    if legacy_feats != CANONICAL_FEATURE_LIST:
        diff_added   = set(CANONICAL_FEATURE_LIST) - set(legacy_feats)
        diff_dropped = set(legacy_feats) - set(CANONICAL_FEATURE_LIST)
        raise ValueError(
            f"feature_list mismatch:\n"
            f"  in canonical, not in legacy: {sorted(diff_added)}\n"
            f"  in legacy, not in canonical: {sorted(diff_dropped)}"
        )

    # ────────── Stacking: один файл list[LGBMRegressor] → 10 файлов ──────────
    stack_pack = joblib.load(legacy_dir / "averaging_ensemble_10lgbm.joblib")
    if not isinstance(stack_pack, list):
        raise TypeError(f"Expected list of 10 models, got {type(stack_pack)}")
    if len(stack_pack) != 10:
        logger.warning("Expected 10 stack models, got %d (proceeding)", len(stack_pack))

    # ────────── Spike компоненты ──────────
    classifier_hi      = joblib.load(legacy_dir / "spike_classifier_high.joblib")
    classifier_lo      = joblib.load(legacy_dir / "spike_classifier_low.joblib")
    regressor_spike_hi = joblib.load(legacy_dir / "spike_regressor_high.joblib")
    regressor_spike_lo = joblib.load(legacy_dir / "spike_regressor_low.joblib")

    # ────────── BlendParams из metadata.spike_blend ──────────
    sb = legacy_meta["spike_blend"]
    blend = BlendParams(
        k_hi   = float(sb["best_k_hi"]),
        w_hi   = float(sb["best_w_hi"]),
        thr_lo = float(sb["best_thr_lo"]),
        w_lo   = float(sb["best_w_lo"]),
    )

    # ────────── FittedFeatureParams из metadata.quantiles + wmax/smax ──────────
    q = legacy_meta["quantiles"]
    feat_params = FittedFeatureParams(
        wind_forecast_max      = float(legacy_meta["wmax"]),
        solar_forecast_max     = float(legacy_meta["smax"]),
        wind_forecast_p20      = float(q["wind_forecast_p20"]),
        solar_forecast_p80     = float(q["solar_forecast_p80"]),
        de_da_p90              = float(q["de_da_p90_train"]),
        solar_radiation_fc_p95 = float(q["solar_radiation_forecast_p95"]),
        fit_period_start = legacy_meta.get("train_period", "").split("→")[0].strip() or "2021-01-31",
        fit_period_end   = legacy_meta.get("train_period", "").split("→")[-1].strip() or "2024-12-31",
    )

    # ────────── TrainingMetadata ──────────
    train_period = legacy_meta.get("train_period", "")
    test_period  = legacy_meta.get("test_period", "")
    metrics      = legacy_meta.get("metrics", {})

    md = TrainingMetadata(
        architecture    = legacy_meta.get("architecture", "AveragingEnsemble10LGBM + AsymmetricSpikeBlend"),
        train_start     = train_period.split("→")[0].strip(),
        train_end       = train_period.split("→")[-1].strip(),
        val_start       = test_period.split("→")[0].strip(),
        val_end         = test_period.split("→")[-1].strip(),
        val_mae         = float(metrics.get("mae", float("nan"))),
        val_rmse        = float(metrics.get("rmse", float("nan"))),
        val_smape       = float(metrics.get("smape", float("nan"))),
        val_naive_mae   = float(metrics.get("naive_mae", float("nan"))),
        improvement_pct = float(metrics.get("improvement_pct", float("nan"))),
        git_sha         = "",
        python_version  = legacy_meta.get("python_version", ""),
        fit_timestamp   = legacy_meta.get("created_at", ""),
        stack_configs   = legacy_meta.get("stack_configs", []),
    )

    # ────────── Собираем bundle ──────────
    bundle = ModelBundle(
        version=NEW_VERSION,
        stack_models=stack_pack,
        classifier_hi=classifier_hi,
        classifier_lo=classifier_lo,
        regressor_spike_hi=regressor_spike_hi,
        regressor_spike_lo=regressor_spike_lo,
        blend_params=blend,
        feature_params=feat_params,
        feature_list=CANONICAL_FEATURE_LIST,
        feature_eng_hash=features_module_hash(),
        metadata=md,
    )

    out_dir = save_bundle(bundle, set_as_current=True)
    logger.info("Migrated %s → %s", LEGACY_VERSION, NEW_VERSION)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = migrate()
    print(f"\nMigrated bundle: {out}")
    print(f"current.txt now points to: {NEW_VERSION}")


if __name__ == "__main__":
    main()
