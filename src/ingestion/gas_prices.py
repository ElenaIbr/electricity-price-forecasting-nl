# ================================
# Download TTF gas prices (daily) via Yahoo Finance
# Ticker: TTF=F
# Period: 2019-01-01 → 2025-12-31
# Purpose: RAW data ingestion
# ================================

from pathlib import Path
import pandas as pd
import yfinance as yf


TICKER = "TTF=F"
START_DATE = "2019-01-01"
END_DATE   = "2025-12-31"

OUTPUT_FILENAME = "ttf_gas_price_daily_2019_2025.csv"


def main() -> None:
    # ---- Project paths ----
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw" / "gas"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_FILENAME

    # ---- Download ----
    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No gas price data downloaded from Yahoo Finance")

    # ---- Keep only Close ----
    df = df.reset_index()
    df = df[["Date", "Close"]]
    df.columns = ["date", "gas_price"]

    # ---- Save RAW ----
    df.to_csv(output_path, index=False)

    print("Gas prices (TTF): OK")
    print(f"Rows: {len(df)} | Range: {df.date.min()} → {df.date.max()}")


if __name__ == "__main__":
    main()
