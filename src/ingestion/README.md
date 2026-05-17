# RAW Data Ingestion

This folder contains ingestion scripts for the diploma project dataset.

The goal of the ingestion layer is to download data from external sources, normalize basic schemas, validate timestamps, and store RAW data in PostgreSQL.

There are **two ingestion modes**:

| Mode | Folder | Purpose | Table prefix | Write semantics |
|------|--------|---------|--------------|-----------------|
| **Historical** (backfill) | `src/ingestion/*.py` | One-shot full history download for training. | `raw_*` | `if_exists="replace"` — full table rewrite |
| **Operational** (daily) | `src/ingestion/operational/` | Incremental pull of a recent window for daily inference. | `op_*` | Delete-by-window + append (idempotent) |

## Rules

- No resampling
- No aggregation
- No feature engineering
- No target creation
- No leakage-prone transformations
- Timestamps must be converted to UTC
- RAW data should stay as close to the original source as possible

Feature construction belongs to a separate transform/processing layer. Yes, even if it is "just one small column". That is how pipelines become swamp creatures.

## Environment variables

Create a `.env` file in the project root:

```bash
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/database_name
ENTSOE_API_TOKEN=your_entsoe_api_token
TENNET_API_KEY=your_tennet_api_key
```

## Operational ingestion (daily inference)

The operational layer is the right entrypoint for any **as-of-now** data pull
that feeds D+1 forecasting. It works on a short window
`[as_of - lookback_days, as_of + forecast_days]`, defined by
`OperationalWindow` in `src/ingestion/base.py`.

```bash
# default: as_of = now UTC, lookback 35d, forecast 2d
python -m src.ingestion.operational.runner

# explicit as_of (useful for replay / debugging)
python -m src.ingestion.operational.runner --as-of 2026-05-10T11:00

# only run a subset of sources
python -m src.ingestion.operational.runner --only weather,gas

# show what would be done without writing to DB
python -m src.ingestion.operational.runner --dry-run
```

Available sources (registered in `operational/runner.py::OPERATIONAL_FETCHERS`):

| Key | Class | Output table |
|-----|-------|--------------|
| `da_prices` | `EntsoeDayAheadPricesOperational` | `op_da_prices_hourly` |
| `load_actual` | `EntsoeLoadActualOperational` | `op_load_actual_15min` |
| `load_forecast` | `EntsoeLoadForecastOperational` | `op_load_forecast_15min` |
| `generation_forecast` | `EntsoeGenerationForecastOperational` | `op_generation_forecast_15min` |
| `weather` | `OpenMeteoWeatherOperational` | `op_weather_hourly` (incl. `kind` ∈ {actual, forecast}) |
| `gas` | `GasTtfDailyOperational` | `op_gas_price_daily` |

Each operational fetcher inherits from `BaseFetcher`:

```python
class MyFetcher(BaseFetcher):
    table_name = "op_my_source"
    def fetch(self, window: OperationalWindow) -> pd.DataFrame: ...
```

`save()` is provided by the base class and uses transactional
**delete-by-window + append**: re-running with the same window yields the
same result; nothing accumulates duplicates.

### Operational timing — what's known when

For NL day-ahead, gate closes at **12:00 CET D-1**, results published
~12:42 CET D-1. So at any operational `as_of`:

| Source | Available range | Used as |
|--------|-----------------|---------|
| NL DA prices | up to end of day(`as_of`) (if past 12:42 D-1) | `lag_1d`, `lag_7d` |
| DE/BE/FR DA prices | same | cross-border lags only |
| Load actual | up to ~`as_of - 1h` | `actual_load_lag_1d` |
| Load forecast | through D+1 | `load_forecast` (direct) |
| Generation forecast (wind/solar) | through D+1 | direct features |
| Weather | past_days actual + 16 forecast days (single API call) | direct + lags |
| Gas (TTF) | yesterday's close | `gas_lag_1d` |

> ⚠️ Cross-border DA prices for D+1 are **not** available at NL inference time
> (other auctions clear simultaneously). Only lagged values are admissible.

## Historical (backfill) orchestrator

For full re-download of historical data (used for training):

`python -m src.pipelines.run_ingestion`

### Data Sources

## ENTSO-E

* Load (actual & forecast)
* Imbalance prices
* Day-ahead prices
* Solar & wind generation (actual & forecast)
* Cross-border flows
* Installed capacity

Typical frequency: 15 min / hourly

## TenneT
* Settlement prices
* FRR activations
* Settled imbalance volumes
* Merit order list
* Balance delta (high-res)

Notes:

* Many endpoints use Europe/Amsterdam timezone
* Converted to UTC during ingestion
* DST handled in tennet_common.py

## Open-Meteo
* Historical actual weather
* Historical weather forecasts

## Yahoo Finance
* CO2 price (EU ETS proxy)
* Gas price (TTF proxy)

Frequency: daily

## ENTSO-E
* entsoe_load_15.py
* entsoe_load_forecast_15.py
* entsoe_imbalance_price_15.py
* entsoe_prices_hourly.py
* entsoe_solar_15.py
* entsoe_solar_forecast_15.py
* entsoe_wind_15.py
* entsoe_wind_forecast_15.py
* entsoe_crossborder_flows_15.py
* entsoe_transfer_capacity_15.py
* entsoe_installed_capacity_15.py

## TenneT
* tennet_settlement_prices.py
* tennet_frr_activations.py
* tennet_settled_imbalance_volumes.py
* tennet_merit_order_list.py
* tennet_balance_delta_high_res.py

## Weather
* open_meteo_weather_actual.py
* open_meteo_weather_forecast.py

## Market data
* yfinance_co2_price.py
* yfinance_gas_price.py

## Pipeline structure
Each ingestion script follows:

```python
def main():
    df = load_data()

    if df.empty:
        raise RuntimeError("No data loaded")

    df = normalize(df)

    save_raw_table(df)
```

## PostgreSQL behavior
Tables are written with:

`if_exists="replace"`

This keeps the dataset reproducible.

In real production, this should be changed to incremental ingestion.

## Important warning
Do NOT do this in ingestion:

```python
df = df.resample("1h").mean()
df["spread"] = df["a"] - df["b"]
df["lag_96"] = df["price"].shift(96)
```

All of that belongs to a separate transform/modeling layer.
