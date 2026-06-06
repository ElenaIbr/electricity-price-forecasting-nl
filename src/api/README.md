# Inference API

FastAPI-сервис для прогноза hourly DA-цен NL.

При старте сервер один раз загружает текущий model bundle (`models/current.txt`)
через `ForecastPipeline.from_bundle("current")` и держит его в памяти.

Operational inference теперь работает через PostgreSQL: API сам читает свежие
operational tables, собирает master frame, строит features и возвращает прогноз.
`history` больше не передаётся в основной forecast endpoint.

## Запуск

```bash
# default: 127.0.0.1:8000
python -m src.api.cli

# кастомные параметры
python -m src.api.cli --host 0.0.0.0 --port 8000 --reload

# или напрямую через uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

После старта:

* интерактивная Swagger-документация: http://localhost:8000/docs
* OpenAPI schema: http://localhost:8000/openapi.json

## Переменные окружения

| Var                   | Default     | Что делает                                                                   |
| --------------------- | ----------- | ---------------------------------------------------------------------------- |
| `MODEL_VERSION`       | `current`   | Какой bundle грузить. `current` читает `models/current.txt`.                 |
| `STRICT_FEATURE_HASH` | `1`         | При несовпадении `feature_eng_hash` поднять exception. `0` — только warning. |
| `DATABASE_URL`        | —           | PostgreSQL connection string для чтения operational данных.                  |
| `API_HOST`            | `127.0.0.1` | Default host для CLI.                                                        |
| `API_PORT`            | `8000`      | Default port для CLI.                                                        |

## Endpoints

### `GET /health`

Liveness + версия модели.

```bash
curl -s http://localhost:8000/health | jq
```

```json
{"status": "ok", "api_version": "0.1.0", "model_version": "v1.0.0-migrated"}
```

---

### `GET /info`

Все метаданные текущего bundle: blend params, fitted feature params, train/val MAE,
feature_eng_hash, и т.д.

```bash
curl -s http://localhost:8000/info | jq '.metadata.val_mae, .blend_params'
```

---

### `GET /info/features`

Список input-колонок master frame (`required_input_columns`) и финальный
`feature_list` модели.

```bash
curl -s http://localhost:8000/info/features | jq
```

---

### `GET /forecast`

Главный operational endpoint.

Не принимает `history`.
API сам читает PostgreSQL operational tables, собирает hourly master frame,
строит features через `build_feature_frame()` и возвращает 24 hourly прогноза.

Default:

* `as_of` = текущий UTC час;
* `target_date` = следующий Amsterdam calendar day;
* `lookback_days` = 60;
* `forecast_days` = 2.

```bash
curl -s "http://localhost:8000/forecast" | jq
```

Для конкретной даты:

```bash
curl -s "http://localhost:8000/forecast?target_date=2026-06-07" | jq
```

С явным `as_of`:

```bash
curl -s "http://localhost:8000/forecast?as_of=2026-06-06T10:00:00Z&target_date=2026-06-07" | jq
```

#### Query params

| Param           | Default                 | Что делает                                                  |
| --------------- | ----------------------- | ----------------------------------------------------------- |
| `target_date`   | tomorrow Amsterdam date | Дата прогноза в формате `YYYY-MM-DD`.                       |
| `as_of`         | current UTC hour        | Момент, из которого делается прогноз.                       |
| `lookback_days` | `60`                    | Сколько дней истории читать из БД для lag/rolling features. |
| `forecast_days` | `2`                     | На сколько дней вперёд читать forecast-источники.           |

#### Response

```json
{
  "target_date": "2026-06-07",
  "model_version": "v1.0.0-migrated",
  "forecast_made_at": "2026-06-06T18:55:00Z",
  "n_hours": 24,
  "hourly": [
    {"timestamp": "2026-06-07T00:00:00Z", "predicted_price": 72.04},
    {"timestamp": "2026-06-07T01:00:00Z", "predicted_price": 72.79}
  ]
}
```

#### Возможные ошибки

* `503 No day-ahead prices found in DB` — operational tables ещё не наполнены.
* `500 missing_master_columns` — master frame не содержит одну из обязательных input-колонок.
* `422 nan_in_features` — после FE фичи target дня содержат NaN. Обычно причина: мало истории, дыры в БД или нет forecast-данных на target date.
* `400 No feature rows for target_date` — в собранном feature frame нет строк на выбранную дату.

---

### `GET /forecast/debug`

То же, что `GET /forecast`, но возвращает компоненты прогноза:
`y_base`, `y_spike_hi`, `y_spike_lo`, `prob_hi`, `prob_lo`, `risk_hi`, `risk_lo`, `y_final`.

```bash
curl -s "http://localhost:8000/forecast/debug?target_date=2026-06-07" | jq
```

Используется для:

* диагностики spike blend;
* мониторинга расхождений `y_base` ↔ `y_final`;
* визуализации debug-компонентов в dashboard.

---

### `POST /forecast/from-history`

Research/debug endpoint.

Работает как старый `/forecast`: принимает hourly
`history` в request body, строит features и возвращает прогноз.

Этот режим нужен для:

* offline experiments;
* unit/integration tests;
* проверки модели на кастомном history dataset;
* сравнения разных master-frame сборок.

```bash
curl -s -X POST "http://localhost:8000/forecast/from-history" \
  -H "Content-Type: application/json" \
  -d @payload.json | jq
```

#### Request body

```json
{
  "history": [
    {
      "timestamp": "2025-04-01T00:00:00Z",
      "nl_day_ahead_price": 80.5,
      "be_day_ahead_price": 78.2,
      "de_day_ahead_price": 79.1,
      "fr_day_ahead_price": 76.8,
      "gas_price": 32.4,
      "net_flow_de_nl": 1200.0,
      "net_flow_be_nl": -300.0,
      "imbalance_price_long": 95.0,
      "imbalance_price_short": 70.0,
      "load_forecast": 11500.0,
      "wind_forecast_mw": 2400.0,
      "solar_forecast_mw": 0.0,
      "temperature_c": 8.5,
      "wind_ms": 5.2,
      "solar_radiation": 0.0,
      "cloud_cover": 75.0,
      "humidity": 82.0,
      "temperature_forecast": 8.6,
      "wind_speed_forecast": 5.4,
      "solar_radiation_forecast": 0.0
    }
  ],
  "as_of": "2025-05-10T11:00:00Z",
  "target_date": "2025-05-11"
}
```

---

### `GET /market/actuals`

Возвращает фактические NL day-ahead цены из PostgreSQL для выбранной Amsterdam calendar date.

Если `target_date` не передан, API берёт последнюю доступную дату в таблице
`op_da_prices_hourly` для `country_label = 'nl'`.

```bash
curl -s "http://localhost:8000/market/actuals" | jq
```

Для конкретной даты:

```bash
curl -s "http://localhost:8000/market/actuals?target_date=2026-06-07" | jq
```

#### Response

```json
{
  "target_date": "2026-06-07",
  "n_hours": 24,
  "hourly": [
    {"timestamp": "2026-06-07T00:00:00Z", "actual_price": 73.10},
    {"timestamp": "2026-06-07T01:00:00Z", "actual_price": 69.80}
  ]
}
```

Если данных за дату нет:

```json
{
  "target_date": "2026-06-07",
  "n_hours": 0,
  "hourly": []
}
```

## Архитектура

```text
src/api/
├── main.py            # FastAPI app, lifespan-загрузка bundle
├── cli.py             # uvicorn launcher
├── dependencies.py    # PipelineDep (DI)
├── schemas.py         # Pydantic v2 модели IO
└── routes/
    ├── health.py      # /health, /info, /info/features
    ├── forecast.py    # GET /forecast, GET /forecast/debug, POST /forecast/from-history
    └── market.py      # GET /market/actuals
```

Operational flow для `GET /forecast`:

```text
PostgreSQL operational tables
    ↓
read_master_frame_from_db(as_of, lookback_days, forecast_days)
    ↓
build_feature_frame(master, pipe.bundle.feature_params, ctx)
    ↓
slice target_date rows
    ↓
pipe.predict(X)
    ↓
ForecastResponse
```

Streamlit flow:

```text
GET /market/actuals?target_date=...
GET /forecast?target_date=...
    ↓
dashboard draws actual + forecast
```

## Что НЕ делает

* **Не делает ingestion внутри API.** Свежие данные тянутся отдельной CLI командой:
  `python -m src.ingestion.operational.runner`.
* **Не переобучает модель.** API только загружает готовый model bundle.
* **Не читает parquet/CSV cache в operational mode.** Источник данных для `/forecast`
  и `/market/actuals` — PostgreSQL.
