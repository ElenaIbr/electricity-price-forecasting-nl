# NL Day-Ahead Electricity Price Forecasting

Production-ready ML система для почасового прогноза day-ahead цен на
электроэнергию в Нидерландах (NL DA market) на горизонт **D+1** (24 часа).

Master's thesis. Прогноз делается **до закрытия day-ahead аукциона**
(12:00 CET D-1) и использует только данные, реально доступные на этот момент.

## Текущие результаты

| Метрика | Значение | Контекст |
|---|---|---|
| **MAE (валидация 2025)** | **14.55 EUR/MWh** | 8 757 часов |
| Naive -7d baseline | 30.11 EUR/MWh | то же окно |
| **Улучшение vs naive** | **51.66%** | |
| RMSE | 23.56 | |
| sMAPE | 32.6% | |
| Live 2026-04-15 (24h, real data) | MAE 9.93 EUR/MWh | live ENTSO-E + Open-Meteo + yfinance |

Архитектура: **Averaging Ensemble (10 diverse LightGBM)** + **Asymmetric
Spike Classifier Blend** (top-k% для HIGH spikes, threshold-based для LOW).

## Архитектура

```
┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌─────────────┐
│ Ingestion   │→ │ Curated DB  │→ │ Feature engineer │→ │ Model bundle │→ │ Inference   │
│             │  │ (Postgres)  │  │ (single contract)│  │ (versioned)  │  │ (CLI / API) │
└─────────────┘  └─────────────┘  └──────────────────┘  └──────────────┘  └─────────────┘
   ENTSO-E         raw_*           build_feature_     stacking +         POST /forecast
   Open-Meteo      op_*            frame()            classifiers +      python -m
   yfinance        (incremental)   FittedFeature-     blend params        inference
                                   Params (квантили)  (joblib)
```

Главный архитектурный принцип — **единый контракт `build_feature_frame()`
между training и inference**. Один и тот же код считает фичи в обучении и
в продакшен-инференсе. Контракт защищён тестами no-leakage и
`feature_eng_hash` (SHA-256 модуля) внутри каждого bundle.

## Структура проекта

```
electricity-price-forecasting-nl/
├── src/
│   ├── ingestion/              # Источники данных
│   │   ├── *.py                # historical (backfill, if_exists=replace)
│   │   ├── operational/        # daily incremental fetchers (op_* tables)
│   │   │   ├── base.py         # BaseFetcher, OperationalWindow, replace_window
│   │   │   ├── entsoe_*.py     # DA prices, load (act+fc), gen forecast
│   │   │   ├── open_meteo_weather.py     # past_days actual + forecast_days
│   │   │   ├── gas_ttf.py
│   │   │   └── runner.py       # CLI: python -m src.ingestion.operational.runner
│   │   └── README.md
│   │
│   ├── features/               # ★ ЕДИНЫЙ КОНТРАКТ training ↔ inference
│   │   ├── feature_spec.py     # TARGET, INPUT_COLUMNS (20), FEATURE_LIST (82)
│   │   ├── params.py           # FittedFeatureParams.fit/save/load (квантили)
│   │   ├── holidays_nl.py      # NL календарь
│   │   ├── build_features.py   # build_feature_frame(history, params, ctx)
│   │   └── tests/test_no_leakage.py  # 6 invariant-тестов
│   │
│   ├── models/                 # Production артефакты
│   │   ├── bundle.py           # ModelBundle, BlendParams, TrainingMetadata
│   │   ├── registry.py         # save_bundle / load_bundle / current pointer
│   │   ├── pipeline.py         # ForecastPipeline.predict / predict_with_components
│   │   ├── spike_blend.py      # asymmetric blend helpers
│   │   ├── migrate_v1.py       # legacy notebook bundle → новый формат
│   │   └── tests/test_bundle_io.py
│   │
│   ├── api/                    # FastAPI inference service
│   │   ├── main.py             # app + lifespan-загрузка bundle
│   │   ├── routes/{health,forecast}.py
│   │   ├── schemas.py          # Pydantic v2
│   │   ├── cli.py              # uvicorn launcher
│   │   └── README.md
│   │
│   ├── pipelines/              # backfill / monolith ingestion entrypoints
│   ├── config/settings.py      # latitude/longitude, date ranges, tickers
│   └── db/connection.py        # SQLAlchemy engine
│
├── notebooks/                  # EDA + research (НЕ в production-flow)
├── models/
│   ├── current.txt             # указатель на активную версию
│   ├── v1.0.0/                 # legacy notebook export
│   └── v1.0.0-migrated/        # новый формат: stacking/, classifier_*, blend_params, ...
├── scripts/
│   └── test_on_2026.py         # one-shot prediction на любую дату 2026
├── docker-compose.yml          # Postgres 16
├── requirements.txt
└── .env                        # ENTSOE_API_TOKEN, DATABASE_URL, TENNET_API_KEY
```

## Quick start

### Установка

```bash
pip install -r requirements.txt
docker compose up -d                   # Postgres
python -c "from src.db.connection import get_engine; print(get_engine())"
```

### Inference на тестовой дате 2026

Самодостаточный скрипт — догружает живые данные из ENTSO-E + Open-Meteo
+ yfinance, склеивает с историей 2021-2025 и запускает прогноз:

```bash
python scripts/test_on_2026.py --target-date 2026-04-15
# → MAE 9.93 EUR/MWh
```

### API server

```bash
python -m src.api.cli                  # http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/info | jq .metadata.val_mae

# 24-hour forecast (см. src/api/README.md для полного payload)
curl -X POST http://localhost:8000/forecast -H 'Content-Type: application/json' -d @payload.json
```

### Operational ingestion (свежие данные за окно)

```bash
python -m src.ingestion.operational.runner --dry-run
python -m src.ingestion.operational.runner --as-of 2026-05-10T11:00 --only weather,gas
python -m src.ingestion.operational.runner            # полный pull
```

### Историческая загрузка (backfill для тренировки)

```bash
python -m src.pipelines.run_ingestion
```

## Operational timing — что и когда известно

NL DA gate closure: **12:00 CET D-1**, результаты публикуются ~12:42 CET D-1.

| Источник | Доступно к as_of | Используется как |
|---|---|---|
| NL DA prices | до конца D-1 | `lag_1d`, `lag_7d`, `roll_*` |
| DE/BE/FR DA prices | до конца D-1 | **только лаги** (D+1 ещё не clear-ed) |
| Load forecast | через D+1 | direct feature |
| Generation forecast (wind/solar) | через D+1 | direct feature |
| Weather actual | up to ~as_of - 1h | lag-only |
| Weather forecast | через D+16 | direct feature |
| Gas TTF | yesterday's close | `gas_lag_1d` |
| Imbalance prices | с задержкой | `imb_*_lag_1d/7d` |

Тест `test_no_future_leakage` гарантирует: фичи в момент T используют
только данные до T. Cross-border DA для D+1 не используются никак, кроме
`*_lag_1d` и `*_lag_7d`.

## Components status

| Слой | Статус | Что есть |
|---|---|---|
| Ingestion (historical) | ✅ Done | 17 скриптов, ENTSO-E + TenneT + Open-Meteo + yfinance |
| Ingestion (operational) | ✅ Done | 6 fetchers + runner + CLI + delete-by-window upsert |
| Features | ✅ Done | `build_feature_frame` + 6 no-leakage tests + `FittedFeatureParams` |
| Model bundle | ✅ Done | save/load/migrate + `feature_eng_hash` integrity check |
| Forecast pipeline | ✅ Done | stack→clf→blend, `predict_with_components` |
| API (inference) | ✅ Done | `/health`, `/info`, `/forecast`, `/forecast/debug` |
| Daily inference pipeline | ⏳ TODO | `src/inference/daily.py`: ingest → features → predict → persist |
| Persistence layer | ⏳ TODO | alembic-миграции (raw / curated / predictions схемы) |
| Training pipeline | ⏳ TODO | рефакторинг 06_hourly_forecast notebook → `src/training/` |

## Тесты

```bash
python -m src.features.tests.test_no_leakage     # 6 invariant tests
python -m src.models.tests.test_bundle_io        # bundle save/load smoke
pytest src/ -v
```

## Sources Overview

### Data Sources

**ENTSO-E Transparency Platform**
- Hourly day-ahead electricity prices for NL/DE/BE/FR
- System load (actual + forecast)
- Generation forecast (wind, solar)
- Imbalance prices, cross-border flows

**Open-Meteo**
- Hourly weather (temperature, wind, solar radiation, cloud cover, humidity)
- D+1 forecast endpoint

**TenneT**
- Settlement prices, FRR activations, settled imbalance volumes, merit-order list

**Yahoo Finance (TTF)**
- Daily natural gas price (TTF futures, marginal generation cost proxy)

**KNMI** (research/EDA only)
- De Bilt station historical observations

### Methodological background

- Classical statistical baselines: ARIMA / SARIMA / SARIMAX
- Time-series decomposition: STL, MSTL
- Machine learning: LightGBM ensembles, asymmetric spike correction
- Reference: Lago et al. (2021) "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms"

## Modul-level documentation

- [src/ingestion/README.md](src/ingestion/README.md) — historical vs operational режимы
- [src/api/README.md](src/api/README.md) — endpoints + curl examples + payload schema

## License & attribution

Master's thesis project. Не для commercial use без согласия автора.
Архитектура и production-ready рефакторинг — Claude Opus.
