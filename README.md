# NL Day-Ahead Electricity Price Forecasting

[English](#english) · [Русский](#русский)

> Hourly day-ahead price forecasting for the Dutch electricity market — master's thesis project with production-grade refactoring.
>
> Почасовой прогноз day-ahead цен на электроэнергию в Нидерландах — дипломный проект с production-grade инфраструктурой.

---

## English

### What this project is

This is a **master's thesis** on forecasting hourly **day-ahead electricity prices** for the Netherlands (NL DA market) on a **D+1 horizon** (24 hours ahead). The forecast must be produced **before the day-ahead auction closes** (12:00 CET D-1) and may use only the information actually available at that point.

The work has two parallel goals:

1. **Research** — compare forecasting approaches (classical SARIMAX, linear regression with feature engineering, gradient boosting, ensemble stacking, spike-aware post-processing) on a real five-year dataset, identify what actually works for this market and why.
2. **Production-readiness** — refactor the resulting model into a working pipeline that can be re-run on live data: versioned model bundles, integrity-checked feature engineering, REST API, backtesting on unseen 2026 data.

### Current results

| Metric | Value | Context |
|---|---|---|
| **MAE on 2025 hold-out (test set)** | **14.62 EUR/MWh** | 8 757 hours |
| Naive -7d baseline | 30.11 EUR/MWh | same window |
| **Improvement vs naive** | **51.7 %** | |
| RMSE | 23.74 | |
| sMAPE | 32.97 % | |
| **Live backtest, May 2026** | **MAE 15.68 EUR/MWh** | 31 days, 744 hours, fresh ENTSO-E / Open-Meteo data |
| Median daily MAE (May 2026) | 12.49 | 71 % of days achieve MAE < 15 |

The backtest on May 2026 — data the model has never seen — confirms that the test-set performance generalises. The one outlier day (1 May 2026, Labour Day in DE/BE/FR, MAE 81.94) pulls the monthly average up by ~2 EUR/MWh; without it the model would track 2025 performance closely. This is documented in [src/features/holidays_nl.py](src/features/holidays_nl.py) and discussed under "Known limitations".

### Architecture

```
┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌─────────────┐
│ Ingestion   │→ │ Curated DB  │→ │ Feature engineer │→ │ Model bundle │→ │ Inference   │
│             │  │ (Postgres)  │  │ (single contract)│  │ (versioned)  │  │ (CLI / API) │
└─────────────┘  └─────────────┘  └──────────────────┘  └──────────────┘  └─────────────┘
   ENTSO-E         raw_*           build_feature_     stacking +         POST /forecast
   Open-Meteo     op_*             frame()            classifiers +      scripts/test_on_2026
   yfinance      (incremental)    FittedFeature-      blend params
                                  Params (quantiles) (joblib)
```

The central design decision is the **single training-inference contract**. The same `build_feature_frame()` function computes features both during training (notebook) and at inference (API). This guarantees the absence of feature drift between train and serve — a well-known production failure mode in tabular ML systems. The contract is hardened with:

- **`feature_eng_hash`** — SHA-256 of the feature module source, embedded in every model bundle. A bundle refuses to load if the current code differs from the version it was trained against.
- **No-leakage tests** — invariants verifying that features at time T use only information available before T (see `src/features/tests/test_no_leakage.py`).

Model architecture: **Averaging Ensemble of 10 diverse LightGBM models** + **Asymmetric Spike-Aware Blend** (LOW-spike correction layer only; HIGH-spike layer trained but disabled because its classifier PR-AUC of 0.47 is too noisy for productive use). The asymmetry is itself a result of the experiments: a symmetric HIGH+LOW blend (EXP-9) makes the model *worse*.

### Project structure

```
electricity-price-forecasting-nl/
├── src/
│   ├── ingestion/              # Data sources
│   │   ├── *.py                # historical backfill (if_exists=replace)
│   │   ├── operational/        # daily incremental fetchers (op_* tables)
│   │   │   ├── base.py         # BaseFetcher, OperationalWindow, replace_window
│   │   │   ├── entsoe_*.py     # DA prices, load (actual+forecast), gen forecast
│   │   │   ├── open_meteo_weather.py     # past_days actual + forecast_days
│   │   │   ├── gas_ttf.py
│   │   │   └── runner.py       # CLI: python -m src.ingestion.operational.runner
│   │   └── README.md
│   │
│   ├── features/               # ★ SINGLE CONTRACT training ↔ inference
│   │   ├── feature_spec.py     # TARGET, INPUT_COLUMNS (20), FEATURE_LIST (82)
│   │   ├── params.py           # FittedFeatureParams.fit/save/load (quantiles)
│   │   ├── holidays_nl.py      # NL calendar + EU neighbour holidays
│   │   ├── build_features.py   # build_feature_frame(history, params, ctx)
│   │   └── tests/test_no_leakage.py  # 6 invariant tests
│   │
│   ├── models/                 # Production artefacts
│   │   ├── bundle.py           # ModelBundle, BlendParams, TrainingMetadata
│   │   ├── registry.py         # save_bundle / load_bundle / current pointer
│   │   ├── pipeline.py         # ForecastPipeline.predict / predict_with_components
│   │   ├── spike_blend.py      # asymmetric blend helpers
│   │   ├── migrate_v1.py       # legacy notebook bundle → new format
│   │   └── tests/test_bundle_io.py
│   │
│   ├── api/                    # FastAPI inference service
│   │   ├── main.py             # app + lifespan-loaded bundle
│   │   ├── routes/{health,forecast}.py
│   │   ├── schemas.py          # Pydantic v2
│   │   ├── cli.py              # uvicorn launcher
│   │   └── README.md
│   │
│   ├── pipelines/              # backfill orchestration
│   ├── config/settings.py      # latitude/longitude, date ranges, tickers
│   └── db/connection.py        # SQLAlchemy engine
│
├── notebooks/                  # EDA + research (NOT in production-flow)
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda_v2.ipynb
│   └── 03_modeling_fixed.ipynb
│
├── models/
│   ├── current.txt             # pointer to active version
│   ├── v1.0.0/                 # legacy notebook export
│   └── v1.0.0-migrated/        # new format: stacking/, classifier_*, blend_params, ...
│
├── scripts/
│   ├── test_on_2026.py         # one-shot prediction on any 2026 date
│   └── backtest_2026.py        # range backtest with daily breakdown
│
├── docker-compose.yml          # Postgres 16
├── requirements.txt
└── .env                        # ENTSOE_API_TOKEN, DATABASE_URL, TENNET_API_KEY
```

### Quick start (smoke test, ~5 minutes)

If you just want to check that the trained model can produce a forecast on live data — **no database, no backfill, no training required**, because the model bundle is checked into the repository:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put your ENTSO-E token into .env (see Build from scratch step 3)

python scripts/test_on_2026.py --target-date 2026-04-15
# → 24 hourly predictions vs actual, MAE 9.93 EUR/MWh
```

The script joins five years of historical data shipped in `data/master_hourly_2021_2025.csv` with live data fetched from ENTSO-E + Open-Meteo + yfinance for the target window.

### Build from scratch

The full pipeline, end-to-end. Estimated time: 1–2 hours, mostly waiting for ENTSO-E API rate limits during the backfill.

#### 1. Prerequisites

- Python 3.11+ (3.14 also tested)
- Docker Desktop (only if you want the Postgres path; CSV-only path works without it)
- Free accounts:
  - **ENTSO-E Transparency Platform** — registration at https://transparency.entsoe.eu, then generate API token via My Account
  - **TenneT Data Platform** — optional, only if you want imbalance data
  - **Open-Meteo** — no registration required
  - **yfinance** — no registration required

#### 2. Clone and prepare the environment

```bash
git clone <repo-url>
cd electricity-price-forecasting-nl

python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Configure secrets

Create `.env` in the project root:

```env
ENTSOE_API_TOKEN=<your ENTSO-E token>
DATABASE_URL=postgresql+psycopg2://electricity_user:electricity_pass@localhost:5432/electricity
TENNET_API_KEY=<optional, only if using imbalance>
```

#### 4. Start Postgres (optional)

```bash
docker compose up -d
python -c "from src.db.connection import get_engine; print(get_engine())"
```

Skip this step if you prefer the CSV-only pipeline.

#### 5. Historical backfill (~30–60 min)

```bash
python -m src.pipelines.run_ingestion
```

This fetches five years of hourly data from all sources. ENTSO-E imposes rate limits, hence the long runtime.

#### 6. Assemble the master frame

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
# Run all cells → produces data/master_hourly_2021_2025.csv
```

#### 7. (Optional) Exploratory data analysis

```bash
jupyter notebook notebooks/02_eda_v2.ipynb
# Run all cells → distribution analysis, time patterns, spike analysis
```

#### 8. Train the model

```bash
jupyter notebook notebooks/03_modeling_fixed.ipynb
# Run all cells (~15–30 min)
# - All experiments EXP-0..EXP-9d
# - Final cell saves bundle to models/v1.0.0-migrated/
```

To save under a new version (instead of overwriting), change `version="v1.0.0-migrated"` to e.g. `version="v1.1.0"` in the `save_model` cell.

#### 9. Activate the new version

```bash
echo "v1.0.0-migrated" > models/current.txt
# or for a new version:
# echo "v1.1.0" > models/current.txt
```

#### 10. Run tests

```bash
pytest src/features/tests/test_no_leakage.py -v
pytest src/models/tests/test_bundle_io.py -v
pytest -v
```

#### 11. Forecast on real 2026 data

```bash
python scripts/test_on_2026.py --target-date 2026-04-15
python scripts/backtest_2026.py --start 2026-05-01 --end 2026-05-31 --save-csv backtest_may.csv
```

#### 12. (Optional) Start the API

```bash
python -m src.api.cli
# Swagger UI at http://localhost:8000/docs
```

### Operational timing — what is known when

NL DA gate closure: **12:00 CET D-1**, results published around 12:42 CET D-1.

| Source | Available by as_of | Used as |
|---|---|---|
| NL DA prices | through end of D-1 | `lag_1d`, `lag_7d`, `roll_*` |
| DE / BE / FR DA prices | through end of D-1 | **lags only** (D+1 not yet cleared) |
| Load forecast | through D+1 | direct feature |
| Generation forecast (wind / solar) | through D+1 | direct feature |
| Weather actual | up to ~as_of - 1h | lag-only |
| Weather forecast | through D+16 | direct feature |
| Gas TTF | yesterday's close | `gas_lag_1d` |
| Imbalance prices | with delay | `imb_*_lag_1d/7d` |

The `test_no_future_leakage` invariant guarantees that features at time T use only data available before T. Cross-border DA prices for D+1 are *not* used at all except via `*_lag_1d` and `*_lag_7d`.

### Known limitations

1. **High-spike events remain the main source of residual error.** The HIGH-spike classifier achieves PR-AUC of only 0.47 and including it in the blend makes the model worse (EXP-9 vs EXP-9d). The current feature set lacks scarcity signals — outages, transmission congestion, balancing market stress — that drive high spikes.
2. **Neighbouring-market holidays are not encoded.** May 1, August 15, November 1, November 11 are holidays in DE / BE / FR but not in NL. On these days the NL market behaves like a holiday through cross-border price arbitrage, but the model has no signal for this. The function `eu_neighbour_holidays()` is implemented in [src/features/holidays_nl.py](src/features/holidays_nl.py) but not yet wired into the feature set — that would require model retraining.
3. **Distribution shift between training and inference periods.** Training data (2021–2024) includes the 2022 European energy crisis with mean price ~242 EUR/MWh; test data (2025) and live inference (2026) are post-crisis with mean ~85 EUR/MWh. The model adapts via lag features, but this is fundamentally hard.

### Component status

| Layer | Status | Description |
|---|---|---|
| Ingestion (historical) | ✅ Done | 17 scripts, ENTSO-E + TenneT + Open-Meteo + yfinance |
| Ingestion (operational) | ✅ Done | 6 fetchers + runner + CLI + delete-by-window upsert |
| Features | ✅ Done | `build_feature_frame` + 6 no-leakage tests + `FittedFeatureParams` |
| Model bundle | ✅ Done | save / load / migrate + `feature_eng_hash` integrity check |
| Forecast pipeline | ✅ Done | stack → clf → blend, `predict_with_components` |
| API (inference) | ✅ Done | `/health`, `/info`, `/forecast`, `/forecast/debug` |
| Daily inference pipeline | ⏳ TODO | `src/inference/daily.py`: ingest → features → predict → persist |
| Persistence layer | ⏳ TODO | alembic migrations (raw / curated / predictions schemas) |
| Training pipeline | ⏳ TODO | refactor notebook → `src/training/` |
| Monitoring / drift detection | ⏳ TODO | live MAE alerting, feature drift checks |

### Tests

```bash
python -m src.features.tests.test_no_leakage     # 6 invariant tests
python -m src.models.tests.test_bundle_io        # bundle save/load smoke
pytest -v                                        # everything
```

### Data sources

**ENTSO-E Transparency Platform**
- Hourly day-ahead electricity prices for NL / DE / BE / FR
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

**KNMI** (research / EDA only)
- De Bilt station historical observations

### Methodological references

- Classical statistical baselines: ARIMA / SARIMA / SARIMAX
- Time-series decomposition: STL, MSTL
- Machine learning: LightGBM ensembles, asymmetric spike correction
- Reference: Lago et al. (2021) *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms.* Applied Energy 293, 116983.

### Module-level documentation

- [src/ingestion/README.md](src/ingestion/README.md) — historical vs operational modes
- [src/api/README.md](src/api/README.md) — endpoints + curl examples + payload schema

### License

Master's thesis project. Not for commercial use without the author's consent.

---

## Русский

### Что это за проект

Это **дипломный проект** по прогнозированию почасовых **day-ahead цен на электроэнергию** для Нидерландов (NL DA market) на горизонт **D+1** (24 часа вперёд). Прогноз должен быть сформирован **до закрытия day-ahead аукциона** (12:00 CET в день D-1) и использует только ту информацию, которая реально доступна на этот момент.

У работы две параллельные цели:

1. **Исследовательская** — сравнить подходы (классические SARIMAX, линейная регрессия на engineered features, градиентный бустинг, stacking ensemble, spike-aware post-processing) на пятилетнем датасете реальных рыночных данных и понять что именно работает на этом рынке и почему.
2. **Production-готовность** — превратить итоговую модель в работающий пайплайн, который можно запускать на live данных: версионированные model bundles, integrity-checked feature engineering, REST API, бэктесты на невидимых для модели данных 2026 года.

Большинство академических работ по forecasting заканчиваются метрикой на test set. Здесь — модель вызывается командой `python scripts/test_on_2026.py --target-date 2026-05-15` и возвращает 24 цены на дате, которой не существовало в момент обучения.

### Текущие результаты

| Метрика | Значение | Контекст |
|---|---|---|
| **MAE на hold-out 2025 (test set)** | **14.55 EUR/MWh** | 8 757 часов |
| Naive -7d baseline | 30.11 EUR/MWh | то же окно |
| **Улучшение vs naive** | **51.7 %** | |
| RMSE | 23.56 | |
| sMAPE | 32.6 % | |
| **Live backtest, май 2026** | **MAE 15.68 EUR/MWh** | 31 день, 744 часа, свежие ENTSO-E / Open-Meteo |
| Median daily MAE (май 2026) | 12.49 | 71 % дней дают MAE < 15 |

Бэктест на мае 2026 — это данные, которых модель никогда не видела — подтверждает что качество с test set обобщается. Один outlier-день (1 мая 2026, Labour Day в DE/BE/FR, MAE 81.94) поднимает месячное среднее примерно на 2 EUR/MWh; без него модель работает почти как на 2025. Этот эффект описан в [src/features/holidays_nl.py](src/features/holidays_nl.py) и в разделе «Известные ограничения».

### Архитектура

```
┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌─────────────┐
│ Ingestion   │→ │ Curated DB  │→ │ Feature engineer │→ │ Model bundle │→ │ Inference   │
│             │  │ (Postgres)  │  │ (single contract)│  │ (versioned)  │  │ (CLI / API) │
└─────────────┘  └─────────────┘  └──────────────────┘  └──────────────┘  └─────────────┘
   ENTSO-E         raw_*           build_feature_     stacking +         POST /forecast
   Open-Meteo     op_*             frame()            classifiers +      scripts/test_on_2026
   yfinance      (incremental)    FittedFeature-      blend params
                                  Params (квантили)  (joblib)
```

Главное архитектурное решение — **единый контракт training ↔ inference**. Один и тот же `build_feature_frame()` считает фичи и в обучении (в notebook), и в продакшен-инференсе (API). Это исключает feature drift между train и serve — известный production-failure mode в табличном ML. Контракт защищён двумя механизмами:

- **`feature_eng_hash`** — SHA-256 модуля features, встроенный в каждый model bundle. Bundle отказывается грузиться если код features изменился относительно версии на которой он обучался.
- **No-leakage тесты** — инварианты проверяющие что фичи в момент T используют только данные доступные до T (`src/features/tests/test_no_leakage.py`).

Архитектура модели: **Averaging Ensemble из 10 diverse LightGBM** + **Asymmetric Spike-Aware Blend** (только LOW-spike correction layer; HIGH-spike обучен, но отключён — его classifier PR-AUC 0.47 слишком зашумлён для продуктивного использования). Сама асимметрия — результат экспериментов: симметричный HIGH+LOW blend (EXP-9) делает модель **хуже** базовой.

### Структура проекта

```
electricity-price-forecasting-nl/
├── src/
│   ├── ingestion/              # Источники данных
│   │   ├── *.py                # historical backfill (if_exists=replace)
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
│   │   ├── holidays_nl.py      # NL календарь + EU neighbour holidays
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
│   ├── pipelines/              # backfill orchestration
│   ├── config/settings.py      # latitude/longitude, date ranges, tickers
│   └── db/connection.py        # SQLAlchemy engine
│
├── notebooks/                  # EDA + research (НЕ в production-flow)
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda_v2.ipynb
│   └── 03_modeling_fixed.ipynb
│
├── models/
│   ├── current.txt             # указатель на активную версию
│   ├── v1.0.0/                 # legacy notebook export
│   └── v1.0.0-migrated/        # новый формат: stacking/, classifier_*, blend_params, ...
│
├── scripts/
│   ├── test_on_2026.py         # one-shot прогноз на любую дату 2026
│   └── backtest_2026.py        # backtest диапазона дат с daily breakdown
│
├── docker-compose.yml          # Postgres 16
├── requirements.txt
└── .env                        # ENTSOE_API_TOKEN, DATABASE_URL, TENNET_API_KEY
```

### Быстрый старт (smoke test, ~5 минут)

Если просто хочешь убедиться что обученная модель умеет делать прогноз на live данных — **без БД, без backfill, без обучения**, потому что model bundle уже в репозитории:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# положи ENTSO-E token в .env (см. «Сборка с нуля», шаг 3)

python scripts/test_on_2026.py --target-date 2026-04-15
# → 24 hourly предсказания и факт, MAE 9.93 EUR/MWh
```

Скрипт сам склеивает пять лет исторических данных из `data/master_hourly_2021_2025.csv` с live данными из ENTSO-E + Open-Meteo + yfinance на нужное окно.

### Сборка с нуля

Полный пайплайн end-to-end. Время: ~1–2 часа, в основном ожидание rate-limit'а ENTSO-E во время backfill.

#### 1. Что должно быть на машине

- Python 3.11+ (3.14 тоже работает)
- Docker Desktop — нужен только для Postgres; CSV-only пайплайн работает без него
- Аккаунты (бесплатные):
  - **ENTSO-E Transparency Platform** — регистрация на https://transparency.entsoe.eu, токен через My Account
  - **TenneT Data Platform** — опционально, только если нужны imbalance данные
  - **Open-Meteo** — без регистрации
  - **yfinance** — без регистрации

#### 2. Клонировать и подготовить окружение

```bash
git clone <repo-url>
cd electricity-price-forecasting-nl

python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Настроить секреты

В корне проекта создай файл `.env`:

```env
ENTSOE_API_TOKEN=<твой ENTSO-E токен>
DATABASE_URL=postgresql+psycopg2://electricity_user:electricity_pass@localhost:5432/electricity
TENNET_API_KEY=<опционально, для imbalance>
```

#### 4. Запустить Postgres (опционально)

```bash
docker compose up -d
python -c "from src.db.connection import get_engine; print(get_engine())"
```

Можно пропустить, если используешь CSV-only пайплайн.

#### 5. Исторический backfill (~30–60 мин)

```bash
python -m src.pipelines.run_ingestion
```

Качает 5 лет почасовых данных из всех источников. ENTSO-E rate-limit'ит, отсюда долгое время выполнения.

#### 6. Собрать master frame

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
# Run all cells → создаст data/master_hourly_2021_2025.csv
```

#### 7. (Опционально) EDA

```bash
jupyter notebook notebooks/02_eda_v2.ipynb
# Run all cells → distribution analysis, temporal patterns, spike analysis
```

#### 8. Обучить модель

```bash
jupyter notebook notebooks/03_modeling_fixed.ipynb
# Run all cells (~15–30 минут)
# - Все эксперименты EXP-0..EXP-9d
# - Последняя ячейка сохраняет bundle в models/v1.0.0-migrated/
```

Если хочешь сохранить новую версию (не перезаписать существующую) — в ячейке `save_model` поменяй `version="v1.0.0-migrated"` на например `version="v1.1.0"`.

#### 9. Активировать новую версию

```bash
echo "v1.0.0-migrated" > models/current.txt
# или для новой версии:
# echo "v1.1.0" > models/current.txt
```

#### 10. Прогнать тесты

```bash
pytest src/features/tests/test_no_leakage.py -v
pytest src/models/tests/test_bundle_io.py -v
pytest -v
```

#### 11. Прогноз на реальных данных 2026

```bash
python scripts/test_on_2026.py --target-date 2026-04-15
python scripts/backtest_2026.py --start 2026-05-01 --end 2026-05-31 --save-csv backtest_may.csv
```

#### 12. (Опционально) Запустить API

```bash
python -m src.api.cli
# Swagger UI на http://localhost:8000/docs
```

### Operational timing — что и когда известно

NL DA gate closure: **12:00 CET D-1**, результаты публикуются ~12:42 CET D-1.

| Источник | Доступно к as_of | Используется как |
|---|---|---|
| NL DA prices | до конца D-1 | `lag_1d`, `lag_7d`, `roll_*` |
| DE / BE / FR DA prices | до конца D-1 | **только лаги** (D+1 ещё не clear-ed) |
| Load forecast | через D+1 | direct feature |
| Generation forecast (wind / solar) | через D+1 | direct feature |
| Weather actual | up to ~as_of - 1h | lag-only |
| Weather forecast | через D+16 | direct feature |
| Gas TTF | вчерашнее закрытие | `gas_lag_1d` |
| Imbalance prices | с задержкой | `imb_*_lag_1d/7d` |

Инвариант `test_no_future_leakage` гарантирует: фичи в момент T используют только данные до T. Cross-border DA цены для D+1 **не используются никак**, кроме `*_lag_1d` и `*_lag_7d`.

### Известные ограничения

1. **High-price spikes остаются главным источником residual error.** HIGH-spike classifier достигает PR-AUC всего 0.47, и его включение в blend ухудшает модель (EXP-9 vs EXP-9d). Текущий feature set не содержит scarcity-сигналов — outages, transmission congestion, balancing market stress — которые драйвят высокие спайки.
2. **Праздники соседних рынков не закодированы.** 1 мая, 15 августа, 1 ноября, 11 ноября — это праздники в DE / BE / FR, но не в NL. В эти дни NL рынок ведёт себя как праздничный из-за cross-border арбитража, но у модели нет соответствующего сигнала. Функция `eu_neighbour_holidays()` реализована в [src/features/holidays_nl.py](src/features/holidays_nl.py), но ещё не подключена к feature set — для этого нужно переобучить модель.
3. **Distribution shift между обучением и инференсом.** Train data (2021–2024) включает энергетический кризис 2022 со средней ценой ~242 EUR/MWh; test 2025 и live inference 2026 находятся в постпиковом режиме со средней ~85 EUR/MWh. Модель адаптируется через lag-фичи, но это принципиально трудно.

### Статус компонентов

| Слой | Статус | Что есть |
|---|---|---|
| Ingestion (historical) | ✅ Done | 17 скриптов, ENTSO-E + TenneT + Open-Meteo + yfinance |
| Ingestion (operational) | ✅ Done | 6 fetchers + runner + CLI + delete-by-window upsert |
| Features | ✅ Done | `build_feature_frame` + 6 no-leakage тестов + `FittedFeatureParams` |
| Model bundle | ✅ Done | save / load / migrate + `feature_eng_hash` integrity check |
| Forecast pipeline | ✅ Done | stack → clf → blend, `predict_with_components` |
| API (inference) | ✅ Done | `/health`, `/info`, `/forecast`, `/forecast/debug` |
| Daily inference pipeline | ⏳ TODO | `src/inference/daily.py`: ingest → features → predict → persist |
| Persistence layer | ⏳ TODO | alembic-миграции (raw / curated / predictions схемы) |
| Training pipeline | ⏳ TODO | рефакторинг notebook → `src/training/` |
| Monitoring / drift detection | ⏳ TODO | live MAE alerting, feature drift checks |

### Тесты

```bash
python -m src.features.tests.test_no_leakage     # 6 invariant tests
python -m src.models.tests.test_bundle_io        # bundle save/load smoke
pytest -v                                         # всё
```

### Источники данных

**ENTSO-E Transparency Platform**
- Почасовые day-ahead цены для NL / DE / BE / FR
- Системная нагрузка (факт + прогноз)
- Generation forecast (wind, solar)
- Imbalance prices, cross-border flows

**Open-Meteo**
- Почасовая погода (температура, ветер, solar radiation, cloud cover, humidity)
- D+1 forecast endpoint

**TenneT**
- Settlement prices, FRR activations, settled imbalance volumes, merit-order list

**Yahoo Finance (TTF)**
- Дневная цена газа (TTF futures, proxy маржинальной электростанции)

**KNMI** (research / EDA only)
- Исторические наблюдения станции De Bilt

### Методологические референсы

- Классические статистические baseline: ARIMA / SARIMA / SARIMAX
- Time-series decomposition: STL, MSTL
- Machine learning: LightGBM ensembles, asymmetric spike correction
- Ключевая ссылка: Lago et al. (2021) *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms.* Applied Energy 293, 116983.

### Документация по модулям

- [src/ingestion/README.md](src/ingestion/README.md) — historical vs operational режимы
- [src/api/README.md](src/api/README.md) — endpoints + curl примеры + payload schema

### Лицензия

Дипломный проект. Не для commercial use без согласия автора.
