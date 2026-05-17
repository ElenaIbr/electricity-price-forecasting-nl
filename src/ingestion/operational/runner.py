"""Operational ingestion runner.

Собирает свежие данные для daily inference в одном окне OperationalWindow.

CLI:
  python -m src.ingestion.operational.runner
  python -m src.ingestion.operational.runner --as-of 2026-05-10T11:00 --lookback-days 35
  python -m src.ingestion.operational.runner --only da_prices,weather

Каждый источник обрабатывается изолированно: если один упал, остальные
всё равно отрабатывают. Итог пишется в audit-таблицу op_ingestion_runs.
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import Engine

from src.db.connection import get_engine
from src.ingestion.base import BaseFetcher, OperationalWindow
from src.ingestion.operational.entsoe_da_prices import EntsoeDayAheadPricesOperational
from src.ingestion.operational.entsoe_generation_forecast import (
    EntsoeGenerationForecastOperational,
)
from src.ingestion.operational.entsoe_load import (
    EntsoeLoadActualOperational,
    EntsoeLoadForecastOperational,
)
from src.ingestion.operational.gas_ttf import GasTtfDailyOperational
from src.ingestion.operational.open_meteo_weather import OpenMeteoWeatherOperational

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Registry of operational fetchers
# ──────────────────────────────────────────────────────────────────────────

FetcherFactory = Callable[[], BaseFetcher]

OPERATIONAL_FETCHERS: dict[str, FetcherFactory] = {
    "da_prices":           EntsoeDayAheadPricesOperational,
    "load_actual":         EntsoeLoadActualOperational,
    "load_forecast":       EntsoeLoadForecastOperational,
    "generation_forecast": EntsoeGenerationForecastOperational,
    "weather":             OpenMeteoWeatherOperational,
    "gas":                 GasTtfDailyOperational,
}


# ──────────────────────────────────────────────────────────────────────────
# Run summary
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SourceResult:
    source: str
    status: str           # "ok" | "failed" | "empty"
    rows: int = 0
    error: str | None = None


@dataclass
class RunSummary:
    run_id: str
    as_of: pd.Timestamp
    started_at: pd.Timestamp
    finished_at: pd.Timestamp | None = None
    results: list[SourceResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    def print(self) -> None:
        print(f"\n{'═' * 78}")
        print(f"OPERATIONAL INGESTION RUN  id={self.run_id}")
        print(f"as_of:    {self.as_of}")
        print(f"started:  {self.started_at}")
        print(f"finished: {self.finished_at}")
        print(f"{'─' * 78}")
        for r in self.results:
            mark = {"ok": "✓", "empty": "·", "failed": "✗"}.get(r.status, "?")
            line = f"  {mark}  {r.source:25s}  {r.status:8s}  rows={r.rows}"
            if r.error:
                line += f"  err={r.error[:60]}"
            print(line)
        print(f"{'─' * 78}")
        print(f"OK: {self.ok_count}  FAILED: {self.fail_count}")
        print(f"{'═' * 78}\n")


# ──────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────

def run_operational(
    engine: Engine,
    window: OperationalWindow,
    only: list[str] | None = None,
) -> RunSummary:
    targets = OPERATIONAL_FETCHERS
    if only:
        unknown = [s for s in only if s not in targets]
        if unknown:
            raise ValueError(
                f"Unknown sources: {unknown}. Available: {sorted(targets)}"
            )
        targets = {k: targets[k] for k in only}

    summary = RunSummary(
        run_id=uuid.uuid4().hex[:12],
        as_of=window.as_of,
        started_at=pd.Timestamp.now(tz="UTC"),
    )

    for source_name, factory in targets.items():
        logger.info("══ source: %s ══", source_name)
        try:
            fetcher = factory()
            n = fetcher.run(engine, window)
            status = "ok" if n > 0 else "empty"
            summary.results.append(SourceResult(source_name, status, rows=n))
        except Exception as exc:
            traceback.print_exc()
            summary.results.append(
                SourceResult(source_name, "failed", error=str(exc))
            )

    summary.finished_at = pd.Timestamp.now(tz="UTC")
    return summary


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="src.ingestion.operational.runner",
        description="Operational ingestion: pull latest data window for inference.",
    )
    p.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="As-of timestamp (default: now UTC). Example: 2026-05-10T11:00",
    )
    p.add_argument(
        "--lookback-days", type=int, default=35,
        help="History window length (default 35).",
    )
    p.add_argument(
        "--forecast-days", type=int, default=2,
        help="Forecast horizon length (default 2).",
    )
    p.add_argument(
        "--only", type=str, default=None,
        help=(
            "Comma-separated list of sources to run. "
            f"Available: {','.join(sorted(OPERATIONAL_FETCHERS))}"
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log what would be done, don't write to DB.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    load_dotenv()

    args = parse_args()

    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    window = OperationalWindow.from_as_of(
        as_of=as_of,
        lookback_days=args.lookback_days,
        forecast_days=args.forecast_days,
    )
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    print(f"\nOperational ingestion — {window}")
    if only:
        print(f"Only: {only}")

    if args.dry_run:
        print("\n[DRY RUN] would run:")
        targets = only or list(OPERATIONAL_FETCHERS)
        for s in targets:
            print(f"  · {s}  -> {OPERATIONAL_FETCHERS[s].__name__}")
        return 0

    engine = get_engine()
    summary = run_operational(engine, window, only=only)
    summary.print()

    return 0 if summary.fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
