import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import (
    LOCAL_TIMEZONE,
    START_DATE_HISTORY,
    END_DATE_HISTORY_EXCLUSIVE,
)

TABLE_NAME = "raw_transfer_capacity_hourly"

BORDERS = {
    "de": ("NL", "DE_LU"),
    "be": ("NL", "BE"),
}


def safe_query(
    client: EntsoePandasClient,
    from_country: str,
    to_country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    try:
        return client.query_net_transfer_capacity(
            from_country,
            to_country,
            start=start,
            end=end,
        )
    except Exception as e:
        print(f"Failed ATC {from_country}->{to_country}: {e}")
        return None


def main() -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    api_token = os.getenv("ENTSOE_API_TOKEN")
    if not api_token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set")

    engine = create_engine(database_url)
    client = EntsoePandasClient(api_key=api_token)

    local_tz = timezone(LOCAL_TIMEZONE)

    start = pd.Timestamp(START_DATE_HISTORY, tz=local_tz)
    end = pd.Timestamp(END_DATE_HISTORY_EXCLUSIVE, tz=local_tz)

    dfs: list[pd.DataFrame] = []

    for name, (from_c, to_c) in BORDERS.items():
        print(f"Downloading ATC NL ↔ {name.upper()}")

        atc_forward = safe_query(client, from_c, to_c, start, end)
        atc_reverse = safe_query(client, to_c, from_c, start, end)

        df = pd.DataFrame()

        if atc_forward is not None:
            df[f"atc_nl_to_{name}"] = atc_forward

        if atc_reverse is not None:
            df[f"atc_{name}_to_nl"] = atc_reverse

        if df.empty:
            print(f"No ATC data for {name}")
            continue

        # ---- Time handling ----
        if df.index.tz is None:
            df.index = df.index.tz_localize(LOCAL_TIMEZONE)

        df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"

        # ---- Check frequency ----
        step_counts = df.index.to_series().diff().value_counts(dropna=True)

        print("\nTime step distribution:")
        print(step_counts.head())

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No ATC data collected")

    df_final = pd.concat(dfs, axis=1).sort_index()
    df_final = df_final[~df_final.index.duplicated(keep="first")]

    # ---- Diagnostics ----
    print("\nFinal dataset:")
    print("Rows:", len(df_final))
    print("Range:", df_final.index.min(), "→", df_final.index.max())
    print("Missing values:")
    print(df_final.isna().sum())

    # ---- Save RAW (NO resampling) ----
    df_final = df_final.reset_index()

    df_final["source"] = "entsoe"
    df_final["frequency"] = "hourly"
    df_final["frequency_note"] = "native_hourly_no_resample"
    df_final["created_at"] = pd.Timestamp.now(tz="UTC")

    df_final.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
    )

    print(f"\nSaved {len(df_final)} rows to {TABLE_NAME}")
    print(f"Range: {df_final['timestamp'].min()} → {df_final['timestamp'].max()}")


if __name__ == "__main__":
    main()
