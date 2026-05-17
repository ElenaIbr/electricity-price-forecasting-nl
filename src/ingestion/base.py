"""Базовый слой operational ingestion.

Контракт:
- BaseFetcher.fetch(window) -> DataFrame   (pure, без I/O в БД)
- BaseFetcher.save(df, engine, window)     (delete-by-window + append insert)
- BaseFetcher.run(engine, window)          (fetch + save)

Все timestamps в UTC. CET используется только на boundaries (auction-time helpers).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Operational time window
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OperationalWindow:
    """Окно для operational ingestion.

    history_start ──── as_of ──── forecast_end
        (далёкое прошлое)        (D+1 forecasts)

    Размер lookback подбирается так, чтобы хватило на самые длинные lag-фичи
    (28 дней) + buffer; forecast_days покрывает горизонт прогноза D+1.
    """

    as_of: pd.Timestamp           # tz-aware UTC
    history_start: pd.Timestamp   # UTC
    forecast_end: pd.Timestamp    # UTC

    @classmethod
    def from_as_of(
        cls,
        as_of: Optional[pd.Timestamp] = None,
        lookback_days: int = 35,
        forecast_days: int = 2,
    ) -> "OperationalWindow":
        if as_of is None:
            as_of = pd.Timestamp.now(tz="UTC").floor("h")
        else:
            as_of = pd.Timestamp(as_of)
            as_of = as_of.tz_localize("UTC") if as_of.tz is None else as_of.tz_convert("UTC")

        return cls(
            as_of=as_of,
            history_start=(as_of - pd.Timedelta(days=lookback_days)).floor("D"),
            forecast_end=(as_of + pd.Timedelta(days=forecast_days)).ceil("D"),
        )

    def to_cet(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Удобно для ENTSO-E клиентов, ожидающих Europe/Amsterdam."""
        return (
            self.history_start.tz_convert("Europe/Amsterdam"),
            self.forecast_end.tz_convert("Europe/Amsterdam"),
        )

    def __str__(self) -> str:
        return (
            f"OperationalWindow(as_of={self.as_of:%Y-%m-%d %H:%M %Z}, "
            f"history={self.history_start:%Y-%m-%d}, fcst_end={self.forecast_end:%Y-%m-%d})"
        )


def auction_close_cet(target_date: pd.Timestamp) -> pd.Timestamp:
    """NL DA gate closure: 12:00 CET D-1 для delivery на target_date (D)."""
    target_date = pd.Timestamp(target_date)
    if target_date.tz is None:
        target_date = target_date.tz_localize("Europe/Amsterdam")
    else:
        target_date = target_date.tz_convert("Europe/Amsterdam")
    d_minus_1 = (target_date - pd.Timedelta(days=1)).normalize()
    return d_minus_1.replace(hour=12, minute=0, second=0, microsecond=0)


# ──────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────

def replace_window(
    engine: Engine,
    table: str,
    df: pd.DataFrame,
    time_column: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    extra_filter_sql: Optional[str] = None,
    extra_params: Optional[dict] = None,
) -> int:
    """Транзакционно удалить окно [start, end] из таблицы и вставить df.

    Идемпотентно: повторный запуск с тем же окном даёт тот же результат.
    Если таблицы нет — будет создана при первом INSERT.

    extra_filter_sql / extra_params позволяют доп. ограничить delete
    (например, по country_code, чтобы не задеть параллельные источники).
    """
    if df is None or df.empty:
        logger.warning("replace_window[%s]: empty DataFrame, nothing to write", table)
        return 0

    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    mask = (df[time_column] >= window_start) & (df[time_column] <= window_end)
    df = df.loc[mask]
    if df.empty:
        logger.warning("replace_window[%s]: nothing within window %s..%s",
                       table, window_start, window_end)
        return 0

    insp = inspect(engine)
    table_exists = insp.has_table(table)

    with engine.begin() as conn:
        if table_exists:
            sql = (
                f'DELETE FROM "{table}" '
                f'WHERE "{time_column}" >= :ws AND "{time_column}" <= :we'
            )
            params: dict = {
                "ws": window_start.to_pydatetime(),
                "we": window_end.to_pydatetime(),
            }
            if extra_filter_sql:
                sql += f" AND ({extra_filter_sql})"
                if extra_params:
                    params.update(extra_params)
            conn.execute(text(sql), params)

        df.to_sql(table, conn, if_exists="append", index=False)

    return len(df)


# ──────────────────────────────────────────────────────────────────────────
# DataFrame normalisation helpers
# ──────────────────────────────────────────────────────────────────────────

def to_utc_indexed(obj, value_name: str) -> pd.DataFrame:
    """Приводит Series/DataFrame с tz-naive или CET индексом к UTC DatetimeIndex.

    Возвращает DataFrame с одной колонкой `value_name`.
    """
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=value_name)
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"Expected 1 column, got {obj.shape[1]}: {list(obj.columns)}")
        df = obj.copy()
        df.columns = [value_name]
    else:
        raise TypeError(f"Unexpected type: {type(obj)}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("Europe/Amsterdam")
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ──────────────────────────────────────────────────────────────────────────
# Base fetcher
# ──────────────────────────────────────────────────────────────────────────

class BaseFetcher(ABC):
    """ABC для operational fetchers.

    Подклассы переопределяют:
      table_name   — имя таблицы в БД
      time_column  — колонка с timestamp (default 'timestamp')
      fetch(...)   — возвращает DataFrame
    """

    table_name: str = ""
    time_column: str = "timestamp"
    extra_filter_sql: Optional[str] = None  # для multi-source таблиц

    @abstractmethod
    def fetch(self, window: OperationalWindow) -> pd.DataFrame:
        ...

    def extra_delete_params(self) -> Optional[dict]:
        return None

    def save(
        self,
        df: pd.DataFrame,
        engine: Engine,
        window: OperationalWindow,
    ) -> int:
        if not self.table_name:
            raise NotImplementedError("table_name not set")
        return replace_window(
            engine=engine,
            table=self.table_name,
            df=df,
            time_column=self.time_column,
            window_start=window.history_start,
            window_end=window.forecast_end,
            extra_filter_sql=self.extra_filter_sql,
            extra_params=self.extra_delete_params(),
        )

    def run(self, engine: Engine, window: OperationalWindow) -> int:
        name = self.__class__.__name__
        logger.info("[%s] start: %s", name, window)
        df = self.fetch(window)
        n = self.save(df, engine, window)
        logger.info("[%s] persisted %d rows -> %s", name, n, self.table_name)
        return n
