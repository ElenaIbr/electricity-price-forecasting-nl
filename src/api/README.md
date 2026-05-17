# Inference API

FastAPI-сервис для прогноза hourly DA-цен NL на D+1.

При старте сервер один раз загружает текущий model bundle (`models/current.txt`)
через `ForecastPipeline.from_bundle("current")` и держит его в памяти —
inference на запросе не перечитывает joblib-файлы.

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
- интерактивная Swagger-документация: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json

### Переменные окружения

| Var | Default | Что делает |
|-----|---------|------------|
| `MODEL_VERSION` | `current` | Какой bundle грузить. `current` читает `models/current.txt`. |
| `STRICT_FEATURE_HASH` | `1` | При несовпадении `feature_eng_hash` поднять exception. `0` — только warning. |
| `API_HOST` | `127.0.0.1` | Default host для CLI. |
| `API_PORT` | `8000` | Default port для CLI. |

## Endpoints

### `GET /health`

Liveness + версия модели.

```bash
curl -s http://localhost:8000/health | jq
```
```json
{"status": "ok", "api_version": "0.1.0", "model_version": "v1.0.0-migrated"}
```

### `GET /info`

Все метаданные текущего bundle: blend params, fitted feature params, train/val MAE,
feature_eng_hash, и т.д. Удобно для дашбордов.

```bash
curl -s http://localhost:8000/info | jq '.metadata.val_mae, .blend_params'
```

### `GET /info/features`

Список ожидаемых input-колонок (`required_input_columns`) и финальный
`feature_list` модели. Поможет клиенту собрать корректный payload.

### `POST /forecast`

Главный endpoint. Принимает hourly history + as_of + target_date,
возвращает 24 hourly прогноза на target_date.

**Request body:**

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
    /* … одна запись на каждый hourly слот, ≥ 35 дней истории + строки D+1 с forecast-колонками */
  ],
  "as_of": "2025-05-10T11:00:00Z",
  "target_date": "2025-05-11"
}
```

**Response:**

```json
{
  "target_date": "2025-05-11",
  "model_version": "v1.0.0-migrated",
  "forecast_made_at": "2026-05-10T18:55:00Z",
  "n_hours": 24,
  "hourly": [
    {"timestamp": "2025-05-11T00:00:00Z", "predicted_price": 72.04},
    {"timestamp": "2025-05-11T01:00:00Z", "predicted_price": 72.79},
    ...
  ]
}
```

**Что должно быть в history:**

| Условие | Зачем |
|---------|-------|
| ≥ 35 дней до as_of | warmup для lag/rolling фич (lag_28d, roll_30d_mean, residual_p90 ...). Меньше — будут NaN, /forecast вернёт 422. |
| Записи на target_date с forecast-колонками | `load_forecast`, `wind_forecast_mw`, `solar_forecast_mw`, `temperature_forecast`, `wind_speed_forecast`, `solar_radiation_forecast` — модель использует их напрямую. Цена target дня (`nl_day_ahead_price`) может отсутствовать. |
| UTC timestamps | `2025-05-11T00:00:00Z` или `2025-05-11T00:00:00+00:00`. |

**Возможные ошибки:**

- `422 missing_columns` — в history нет требуемой input-колонки. Проверьте `/info/features`.
- `422 nan_in_features` — после FE фичи target дня содержат NaN. Обычно это короткая история (< 35 дней warmup) или дыры в данных.
- `400 No feature rows for target_date` — нет ни одной строки на target_date в history.

### `POST /forecast/debug`

Тот же `/forecast`, но вместе с финалом `y_final` возвращает все промежуточные
сигналы. Полезно для:

- мониторинга расхождений `y_base` ↔ `y_final`,
- алертов на странные `prob_hi` / `prob_lo`,
- визуализации работы spike blend в дашборде.

**Response (один час):**

```json
{
  "timestamp": "2025-05-11T12:00:00Z",
  "y_base":     -66.58,
  "y_spike_hi": 141.78,
  "y_spike_lo":  -47.27,
  "prob_hi":      0.013,
  "prob_lo":      0.997,
  "risk_hi":      0.000,
  "risk_lo":      0.993,
  "y_final":     -49.32
}
```

Здесь видно: classifier_lo даёт prob_lo = 0.997, что выше threshold (0.567), —
поэтому risk_lo ≈ 1 и y_final подтянут от y_base = −66.58 к y_spike_lo = −47.27.

## Архитектура (resp. модули)

```
src/api/
├── main.py            # FastAPI app, lifespan-загрузка bundle
├── cli.py             # uvicorn launcher
├── dependencies.py    # PipelineDep (DI)
├── schemas.py         # Pydantic v2 модели IO
└── routes/
    ├── health.py      # /health, /info, /info/features
    └── forecast.py    # POST /forecast, POST /forecast/debug
```

Внутри `/forecast` route:
1. `_records_to_df(req.history)` → `pd.DataFrame` с UTC DatetimeIndex
2. валидация INPUT_COLUMNS
3. `build_feature_frame(df, pipe.bundle.feature_params, ctx)` → 82-колоночные фичи
4. срез по `target_date` → 24 строки
5. `pipe.predict(X)` → 24 прогноза
6. сериализация в `ForecastResponse`

## Что НЕ делает (намеренно)

- **Не ходит в БД.** Клиент сам предоставляет history. Daily inference pipeline
  с persistence в Postgres — это `src/inference/daily.py` (пока не реализован).
- **Не делает ingestion.** Свежие данные тянутся отдельным CLI (`python -m src.ingestion.operational.runner`).
- **Не переобучает модель.** Training — отдельный flow в `src/training/` (пока не реализован);
  деплой новой версии = миграция `models/current.txt`.
- **Нет auth.** В production стоит закрыть как минимум `/forecast` через API-key
  middleware или reverse-proxy (nginx + basic auth).
