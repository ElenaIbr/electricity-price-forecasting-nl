"""Health / info endpoints — диагностика без forecast логики."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from src.api.dependencies import PipelineDep
from src.api.schemas import (
    BlendParamsOut,
    FeaturesInfoResponse,
    FittedFeatureParamsOut,
    HealthResponse,
    InfoResponse,
    TrainingMetadataOut,
)
from src.features.feature_spec import FEATURE_LIST, INPUT_COLUMNS

router = APIRouter(tags=["health"])

API_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
def health(pipe: PipelineDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        model_version=pipe.bundle.version,
    )


@router.get("/info", response_model=InfoResponse)
def info(pipe: PipelineDep) -> InfoResponse:
    b = pipe.bundle
    return InfoResponse(
        model_version=b.version,
        feature_eng_hash=b.feature_eng_hash,
        n_stack_models=len(b.stack_models),
        n_features=len(b.feature_list),
        blend_params=BlendParamsOut(**asdict(b.blend_params)),
        feature_params=FittedFeatureParamsOut(**asdict(b.feature_params)),
        metadata=TrainingMetadataOut(
            architecture    = b.metadata.architecture,
            train_start     = b.metadata.train_start,
            train_end       = b.metadata.train_end,
            val_start       = b.metadata.val_start,
            val_end         = b.metadata.val_end,
            val_mae         = b.metadata.val_mae,
            val_naive_mae   = b.metadata.val_naive_mae,
            improvement_pct = b.metadata.improvement_pct,
            git_sha         = b.metadata.git_sha or "",
            fit_timestamp   = b.metadata.fit_timestamp or "",
        ),
    )


@router.get("/info/features", response_model=FeaturesInfoResponse)
def features_info() -> FeaturesInfoResponse:
    """Список ожидаемых input-колонок для /forecast и финальный список фич."""
    return FeaturesInfoResponse(
        required_input_columns=list(INPUT_COLUMNS),
        feature_list=list(FEATURE_LIST),
    )
