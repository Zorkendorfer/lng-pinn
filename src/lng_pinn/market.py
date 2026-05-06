"""ENTSO-E Lithuanian day-ahead price ingestion."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

RAW_DIR = Path("data/raw")
ZONE = "LT"  # ENTSO-E bidding zone


def _token() -> str:
    token = os.environ.get("ENTSOE_API_TOKEN", "")
    if not token:
        raise RuntimeError("Set ENTSOE_API_TOKEN in your .env file.")
    return token


def pull_da_prices(start: str, end: str, zone: str = ZONE) -> pd.DataFrame:
    """Pull day-ahead prices from ENTSO-E and cache to parquet.

    Args:
        start: ISO date string, e.g. "2021-01-01".
        end:   ISO date string, e.g. "2024-01-01".
        zone:  ENTSO-E bidding zone code.

    Returns:
        DataFrame with DatetimeTZDtype index (UTC) and column "price_eur_mwh".
    """
    cache_path = RAW_DIR / f"da_prices_{zone}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    client = EntsoePandasClient(api_key=_token())
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    series = client.query_day_ahead_prices(zone, start=ts_start, end=ts_end)
    df = series.to_frame(name="price_eur_mwh")
    df.index.name = "utc_time"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df
