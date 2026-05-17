"""Model registry: save_bundle / load_bundle / current pointer.

Disk layout:
    models/
        current.txt              # содержит "v1.2.0"
        v1.0.0/
            stacking/
                model_00.joblib
                model_01.joblib
                ...
                model_09.joblib
            classifier_hi.joblib
            classifier_lo.joblib
            regressor_spike_hi.joblib
            regressor_spike_lo.joblib
            blend_params.json
            feature_params.json
            feature_list.json
            metadata.json
            MANIFEST.json        # version, feature_eng_hash, n_stack_models
        v1.1.0/
            ...

Atomicity: записываем во временную папку models/.tmp_{version}/ и
переименовываем в финальное имя одним rename — чтобы не получить
полу-записанный bundle при падении.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import joblib

from src.features.build_features import features_module_hash
from src.features.params import FittedFeatureParams
from src.models.bundle import (
    BlendParams,
    BundleHashMismatchError,
    BundleSchemaError,
    ModelBundle,
    TrainingMetadata,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """src/models/registry.py → project root."""
    return Path(__file__).resolve().parents[2]


MODELS_DIR = _project_root() / "models"
CURRENT_POINTER = MODELS_DIR / "current.txt"


def _bundle_dir(version: str) -> Path:
    return MODELS_DIR / version


# ──────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────

def save_bundle(bundle: ModelBundle, set_as_current: bool = True) -> Path:
    """Сохраняет bundle на диск атомарно.

    Возвращает путь к финальной папке models/{version}/.
    """
    final_dir = _bundle_dir(bundle.version)
    if final_dir.exists():
        raise FileExistsError(
            f"Version {bundle.version} already exists at {final_dir}. "
            f"Bump version or delete manually before re-saving."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Атомарный write: tmp dir → rename
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp_{bundle.version}_", dir=MODELS_DIR))
    try:
        # 1. Stacking ensemble
        stacking_dir = tmp / "stacking"
        stacking_dir.mkdir()
        for i, model in enumerate(bundle.stack_models):
            joblib.dump(model, stacking_dir / f"model_{i:02d}.joblib")

        # 2. Spike компоненты
        joblib.dump(bundle.classifier_hi,      tmp / "classifier_hi.joblib")
        joblib.dump(bundle.classifier_lo,      tmp / "classifier_lo.joblib")
        joblib.dump(bundle.regressor_spike_hi, tmp / "regressor_spike_hi.joblib")
        joblib.dump(bundle.regressor_spike_lo, tmp / "regressor_spike_lo.joblib")

        # 3. JSON компоненты
        (tmp / "blend_params.json").write_text(
            json.dumps(asdict(bundle.blend_params), indent=2, sort_keys=True)
        )
        bundle.feature_params.save(tmp / "feature_params.json")
        (tmp / "feature_list.json").write_text(
            json.dumps(bundle.feature_list, indent=2, ensure_ascii=False)
        )
        (tmp / "metadata.json").write_text(
            json.dumps(asdict(bundle.metadata), indent=2, default=str)
        )

        # 4. MANIFEST — для быстрой проверки совместимости
        manifest = {
            "version":          bundle.version,
            "feature_eng_hash": bundle.feature_eng_hash,
            "n_stack_models":   len(bundle.stack_models),
            "n_features":       len(bundle.feature_list),
        }
        (tmp / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )

        # Атомарный rename
        os.rename(tmp, final_dir)
        logger.info("Saved bundle %s -> %s", bundle.version, final_dir)

    except Exception:
        # Чистим tmp при падении
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        raise

    if set_as_current:
        update_current(bundle.version)

    return final_dir


# ──────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────

def _resolve_version(version: str) -> str:
    if version == "current":
        if not CURRENT_POINTER.exists():
            raise FileNotFoundError(f"{CURRENT_POINTER} does not exist")
        actual = CURRENT_POINTER.read_text().strip()
        if not actual:
            raise ValueError(f"{CURRENT_POINTER} is empty")
        return actual
    return version


def load_bundle(version: str = "current", strict_hash: bool = True) -> ModelBundle:
    """Загружает bundle из models/{version}/.

    strict_hash=True (default): сравнивает feature_eng_hash в MANIFEST с
    текущим хэшем src/features/. При расхождении — BundleHashMismatchError.
    Это защита от silent train/serve skew после изменения FE кода.

    strict_hash=False: загружает в любом случае, только логирует WARNING.
    Использовать в migration / debugging, не в production.
    """
    version = _resolve_version(version)
    bundle_dir = _bundle_dir(version)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    try:
        manifest = json.loads((bundle_dir / "MANIFEST.json").read_text())
    except FileNotFoundError as exc:
        raise BundleSchemaError(f"MANIFEST.json missing in {bundle_dir}") from exc

    saved_hash = manifest.get("feature_eng_hash", "")
    current_hash = features_module_hash()
    if saved_hash != current_hash:
        msg = (
            f"feature_eng_hash mismatch in bundle {version}:\n"
            f"  saved:   {saved_hash}\n"
            f"  current: {current_hash}\n"
            f"FE code изменилось с момента тренировки этой модели."
        )
        if strict_hash:
            raise BundleHashMismatchError(msg)
        logger.warning(msg)

    # 1. Stacking
    stacking_dir = bundle_dir / "stacking"
    if not stacking_dir.exists():
        raise BundleSchemaError(f"stacking/ subdir missing in {bundle_dir}")
    stack_files = sorted(stacking_dir.glob("model_*.joblib"))
    if not stack_files:
        raise BundleSchemaError(f"No stacking models in {stacking_dir}")
    stack_models = [joblib.load(f) for f in stack_files]

    # 2. Spike компоненты
    classifier_hi      = joblib.load(bundle_dir / "classifier_hi.joblib")
    classifier_lo      = joblib.load(bundle_dir / "classifier_lo.joblib")
    regressor_spike_hi = joblib.load(bundle_dir / "regressor_spike_hi.joblib")
    regressor_spike_lo = joblib.load(bundle_dir / "regressor_spike_lo.joblib")

    # 3. JSON
    blend_dict   = json.loads((bundle_dir / "blend_params.json").read_text())
    feature_list = json.loads((bundle_dir / "feature_list.json").read_text())
    metadata     = TrainingMetadata(
        **json.loads((bundle_dir / "metadata.json").read_text())
    )
    feature_params = FittedFeatureParams.load(bundle_dir / "feature_params.json")

    return ModelBundle(
        version=version,
        stack_models=stack_models,
        classifier_hi=classifier_hi,
        classifier_lo=classifier_lo,
        regressor_spike_hi=regressor_spike_hi,
        regressor_spike_lo=regressor_spike_lo,
        blend_params=BlendParams(**blend_dict),
        feature_params=feature_params,
        feature_list=feature_list,
        feature_eng_hash=saved_hash,
        metadata=metadata,
    )


# ──────────────────────────────────────────────────────────────────────────
# Current pointer
# ──────────────────────────────────────────────────────────────────────────

def update_current(version: str) -> None:
    if not _bundle_dir(version).exists():
        raise FileNotFoundError(f"Cannot point to non-existing version {version}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_POINTER.write_text(version + "\n")
    logger.info("models/current.txt -> %s", version)


def list_versions() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(
        d.name for d in MODELS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def current_version() -> Optional[str]:
    if not CURRENT_POINTER.exists():
        return None
    v = CURRENT_POINTER.read_text().strip()
    return v or None
