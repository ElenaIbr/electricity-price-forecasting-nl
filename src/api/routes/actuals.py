"""Market data API endpoints.

GET /market/actuals
    Returns actual NL day-ahead prices from PostgreSQL.

This endpoint is used by Streamlit to draw historical/factual prices
for the selected calendar date.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

from src.db.connection import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["market"], prefix="/market")

LOCAL_TZ = "Europe/Amsterdam"


class HourlyActual(BaseModel):
    timestamp: datetime
    actual_price: float


class ActualsResponse(BaseModel):
    target_date: date
    n_hours: int
    hourly: list[HourlyActual]


def _target_bounds_utc(target_date: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert Amsterdam calendar day to UTC interval."""
    day_start_local = pd.Timestamp(target_date, tz=LOCAL_TZ)
    day_end_local = day_start_local + pd.Timedelta(days=1)

    day_start_utc = day_start_local.tz_convert("UTC")
    day_end_utc = day_end_local.tz_convert("UTC") - pd.Timedelta(seconds=1)

    return day_start_utc, day_end_utc


@router.get("/actuals", response_model=ActualsResponse)
def actuals(
    target_date: date | None = Query(
        default=None,
        description=(
            "Amsterdam calendar date. "
            "If omitted, the latest available NL day-ahead date is used."
        ),
    ),
) -> ActualsResponse:
    engine = get_engine()

    if target_date is None:
        latest_df = pd.read_sql(
            text("""
                SELECT MAX(timestamp) AS max_ts
                FROM op_da_prices_hourly
                WHERE country_label = 'nl'
            """),
            engine,
        )

        max_ts = latest_df.loc[0, "max_ts"]

        if pd.isna(max_ts):
            raise HTTPException(
                status_code=404,
                detail="No NL day-ahead prices found in DB",
            )

        max_ts = pd.Timestamp(max_ts, tz="UTC")
        target_date = max_ts.tz_convert(LOCAL_TZ).date()

    start_utc, end_utc = _target_bounds_utc(target_date)

    df = pd.read_sql(
        text("""
            SELECT timestamp, price_eur_mwh
            FROM op_da_prices_hourly
            WHERE country_label = 'nl'
              AND timestamp >= :start_utc
              AND timestamp <= :end_utc
            ORDER BY timestamp
        """),
        engine,
        params={
            "start_utc": start_utc,
            "end_utc": end_utc,
        },
    )

    if df.empty:
        return ActualsResponse(
            target_date=target_date,
            n_hours=0,
            hourly=[],
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    return ActualsResponse(
        target_date=target_date,
        n_hours=len(df),
        hourly=[
            HourlyActual(
                timestamp=row.timestamp.to_pydatetime(),
                actual_price=float(row.price_eur_mwh),
            )
            for row in df.itertuples(index=False)
        ],
    )
