# RAW Data Ingestion

This folder contains ingestion scripts for the diploma project dataset.

The goal of the ingestion layer is to download data from external sources, normalize basic schemas, validate timestamps, and store RAW data in PostgreSQL.

## Rules

- No resampling
- No aggregation
- No feature engineering
- No target creation
- No leakage-prone transformations
- Timestamps must be converted to UTC
- RAW data should stay as close to the original source as possible

Feature construction belongs to a separate transform/processing layer. Yes, even if it is “just one small column”. That is how pipelines become swamp creatures.

## Environment variables

Create a `.env` file in the project root:

```bash
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/database_name
ENTSOE_API_TOKEN=your_entsoe_api_token
TENNET_API_KEY=your_tennet_api_key
```

## Main orchestrator

Run all ingestion jobs from the project root:

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
