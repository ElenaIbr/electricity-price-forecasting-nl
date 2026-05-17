"""FastAPI app для inference DA-цен NL.

Запуск:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
или:
    python -m src.api.cli
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import forecast as forecast_routes
from src.api.routes import health as health_routes
from src.models.pipeline import ForecastPipeline

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Lifespan: загружаем bundle один раз при старте, отпускаем при shutdown
# ──────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    version = os.environ.get("MODEL_VERSION", "current")
    strict_hash = os.environ.get("STRICT_FEATURE_HASH", "1") not in {"0", "false", "no"}

    logger.info("Loading bundle version=%s (strict_hash=%s)...", version, strict_hash)
    try:
        pipeline = ForecastPipeline.from_bundle(version, strict_hash=strict_hash)
        app.state.pipeline = pipeline
        b = pipeline.bundle
        logger.info(
            "Bundle loaded: %s  val_mae=%.2f  features=%d  stack_models=%d",
            b.version, b.metadata.val_mae, len(b.feature_list), len(b.stack_models),
        )
    except Exception:
        logger.exception("Failed to load bundle. /forecast endpoints will return 503.")
        app.state.pipeline = None

    yield

    app.state.pipeline = None
    logger.info("Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NL Day-Ahead Electricity Price Forecasting",
    description=(
        "Inference API для прогноза hourly DA-цен NL на D+1.\n\n"
        "Endpoints:\n"
        "* `GET /health` — liveness + версия модели\n"
        "* `GET /info` — метаданные модели (MAE, blend params, hash и т.д.)\n"
        "* `GET /info/features` — ожидаемые input/feature колонки\n"
        "* `POST /forecast` — 24 hourly прогноза на target_date\n"
        "* `POST /forecast/debug` — то же + промежуточные сигналы\n"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — для локальной разработки. В production стоит ограничить.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_routes.router)
app.include_router(forecast_routes.router)
