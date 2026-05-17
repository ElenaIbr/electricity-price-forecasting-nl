"""Operational ingestion: pulls latest data window for inference.

В отличие от исторических скриптов рядом, эти fetchers:
  • работают на короткое окно [as_of - lookback, as_of + forecast_days];
  • upsert в таблицы op_* (delete-by-window + append insert);
  • могут запускаться ежедневно через CLI runner;
  • НЕ затирают историю при повторном запуске.

Точка входа — `python -m src.ingestion.operational.runner`.
"""
