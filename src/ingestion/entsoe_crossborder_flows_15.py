import os
import pandas as pd
from entsoe import EntsoePandasClient
from pytz import timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config.settings import (
    LOCAL_TIMEZONE,
    ENTSOE_START_DATE,
    ENTSOE_END_DATE,
)


TABLE_NAME = "raw_crossborder_flows_hourly"

BORDERS = {
    "de": ("NL", "DE_LU"),
    "be": ("NL", "BE"),
}


def to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize(LOCAL_TIMEZONE)

    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df.sort_index()


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

    start = pd.Timestamp(ENTSOE_START_DATE, tz=local_tz)
    end = pd.Timestamp(ENTSOE_END_DATE, tz=local_tz)

    dfs: list[pd.DataFrame] = []

    for name, (from_c, to_c) in BORDERS.items():
        print(f"Downloading flow NL ↔ {name.upper()}")

        flow_out = client.query_crossborder_flows(
            from_c,
            to_c,
            start=start,
            end=end,
        )

        flow_in = client.query_crossborder_flows(
            to_c,
            from_c,
            start=start,
            end=end,
        )

        df = pd.DataFrame({
            f"flow_nl_to_{name}": flow_out,
            f"flow_{name}_to_nl": flow_in,
        })

        df = to_utc_index(df)
        df = df[~df.index.duplicated(keep="first")]

        step_counts = df.index.to_series().diff().value_counts(dropna=True)

        print(f"\n{name.upper()} time step distribution:")
        print(step_counts.head(10))

        df[f"net_flow_{name}_nl"] = (
            df[f"flow_{name}_to_nl"] - df[f"flow_nl_to_{name}"]
        )

        dfs.append(df)

    df_final = pd.concat(dfs, axis=1).sort_index()
    df_final = df_final[~df_final.index.duplicated(keep="first")]
    df_final.index.name = "timestamp"

    print("\nFinal diagnostics:")
    print("Rows:", len(df_final))
    print("Range:", df_final.index.min(), "→", df_final.index.max())
    print("Missing values:")
    print(df_final.isna().sum())

    df_final = df_final.reset_index()
    df_final["source"] = "entsoe"
    df_final["frequency"] = "native_or_mixed"
    df_final["frequency_note"] = "no_resample_raw_crossborder_flows"
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
