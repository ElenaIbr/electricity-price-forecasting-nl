"""FastAPI dependency injection helpers.

The ForecastPipeline is loaded once on startup and stored in app.state.pipeline.

This avoids loading the ~16 MB model bundle on every request,
which would significantly slow down inference.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from src.models.pipeline import ForecastPipeline

logger = logging.getLogger(__name__)


def get_pipeline(request: Request) -> ForecastPipeline:
    """Возвращает pipeline, поднятый при startup."""
    pipe = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Server is starting up or model is unavailable.",
        )
    return pipe


PipelineDep = Annotated[ForecastPipeline, Depends(get_pipeline)]
