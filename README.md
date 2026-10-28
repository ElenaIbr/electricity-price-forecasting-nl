# NL Day-Ahead Electricity Price Forecasting

> Hourly day-ahead price forecasting for the Dutch electricity market — master's thesis project with production-grade refactoring.

---

## English

### What this project is

This is a **master's thesis** on forecasting hourly **day-ahead electricity prices** for the Netherlands (NL DA market) on a **D+1 horizon** (24 hours ahead). The forecast must be produced **before the day-ahead auction closes** (12:00 CET D-1) and may use only the information actually available at that point.

The work has two parallel goals:

1. **Research** — compare forecasting approaches (classical SARIMAX, linear regression with feature engineering, gradient boosting, ensemble stacking, spike-aware post-processing) on a real five-year dataset, identify what actually works for this market and why.
2. **Production-readiness** — refactor the resulting model into a working pipeline that can be re-run on live data: versioned model bundles, integrity-checked feature engineering, REST API, backtesting on unseen 2026 data.

### Current results

| Quality metric | Value |
|---|---|
| MAE | 14.62 EUR/MWh |
| RMSE | 23.74 EUR/MWh |
| sMAPE | 32.97 % |
| Improvement vs Naive-7d baseline | +51.7 % |

### Architecture

```
┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌─────────────┐
│ Ingestion   │→ │ Curated DB  │→ │ Feature engineer │→ │ Model bundle │→ │ Inference   │
│             │  │ (Postgres)  │  │ (single contract)│  │ (versioned)  │  │ (API/UI)    │
└─────────────┘  └─────────────┘  └──────────────────┘  └──────────────┘  └─────────────┘
   ENTSO-E         op_*             build_feature_      stacking +         GET /forecast
   Open-Meteo      raw_*            frame()             classifiers +      GET /market/actuals
   yfinance        (incremental)    FittedFeature-      blend params       Streamlit dashboard
                                    Params              (joblib)
```

The central design decision is the **single training-inference contract**. The same `build_feature_frame()` function computes features both during training (notebook) and at inference (API). This guarantees the absence of feature drift between train and serve — a well-known production failure mode in tabular ML systems. The contract is hardened with:

- **`feature_eng_hash`** — SHA-256 of the feature module source, embedded in every model bundle. A bundle refuses to load if the current code differs from the version it was trained against.
- **No-leakage tests** — invariants verifying that features at time T use only information available before T (see `src/features/tests/test_no_leakage.py`).

Model architecture: **Averaging Ensemble of 10 diverse LightGBM models** + **Asymmetric Spike-Aware Blend** (LOW-spike correction layer only; HIGH-spike layer trained but disabled because its classifier PR-AUC of 0.47 is too noisy for productive use). The asymmetry is itself a result of the experiments: a symmetric HIGH+LOW blend (EXP-9) makes the model *worse*.

### Project structure

```
electricity-price-forecasting-nl/
├── app/streamlit_app.py
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
│   │   ├── routes/{health,forecast, actual}.py
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
| API (inference) | ✅ Done |  GET `/forecast`, GET `/forecast/debug`, GET `/market/actuals`, FastAPI + Streamlit integration |
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
