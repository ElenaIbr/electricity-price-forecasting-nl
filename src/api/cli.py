"""CLI launcher: `python -m src.api.cli`.

Тонкий wrapper вокруг uvicorn с разумными defaults.
"""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="src.api.cli",
        description="Run the NL DA forecasting API server.",
    )
    p.add_argument("--host", default=os.getenv("API_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    p.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only).")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--log-level", default="info")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = parse_args()
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
