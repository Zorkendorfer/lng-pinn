"""ENTSO-E Lithuanian day-ahead price ingestion and weather data fetching."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from entsoe import EntsoePandasClient

RAW_DIR = Path("data/raw")
ZONE = "LT"  # ENTSO-E bidding zone

# Independence FSRU, Klaipėda
LAT = 55.71
LON = 21.13


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


def load_da_prices(start: str, end: str, zone: str = ZONE) -> pd.DataFrame:
    """Load and concatenate cached year-by-year price parquets into one DataFrame.

    Expects files named da_prices_{zone}_{year}-01-01_{year+1}-01-01.parquet
    to exist in data/raw/ (written by pull_da_prices).
    """
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    frames = []
    for year in range(start_year, end_year):
        path = RAW_DIR / f"da_prices_{zone}_{year}-01-01_{year + 1}-01-01.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python scripts/01_pull_entsoe.py "
                f"--start {year}-01-01 --end {year + 1}-01-01"
            )
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def pull_weather(start: str, end: str) -> pd.DataFrame:
    """Fetch hourly T_amb and T_sw for Klaipėda from Open-Meteo and cache.

    Uses ERA5 reanalysis (archive API) for air temperature and the Marine
    API for sea surface temperature.

    Returns:
        DataFrame indexed by UTC hour with columns T_amb (K) and T_sw (K).
    """
    cache_path = RAW_DIR / f"weather_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Open-Meteo end date is inclusive; subtract one day
    end_date = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Air temperature (ERA5 reanalysis)
    r_air = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": start,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": "UTC",
        },
        timeout=60,
    )
    r_air.raise_for_status()
    air = r_air.json()["hourly"]
    t_amb = pd.Series(
        air["temperature_2m"],
        index=pd.to_datetime(air["time"], utc=True),
        name="T_amb",
    )

    # Sea surface temperature (Marine API)
    r_sea = requests.get(
        "https://marine-api.open-meteo.com/v1/marine",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": start,
            "end_date": end_date,
            "hourly": "sea_surface_temperature",
            "timezone": "UTC",
        },
        timeout=60,
    )
    r_sea.raise_for_status()
    sea = r_sea.json()["hourly"]
    t_sw = pd.Series(
        sea["sea_surface_temperature"],
        index=pd.to_datetime(sea["time"], utc=True),
        name="T_sw",
    )

    df = pd.concat([t_amb, t_sw], axis=1)
    # Convert °C → K
    df["T_amb"] = df["T_amb"] + 273.15
    df["T_sw"] = df["T_sw"] + 273.15
    # Forward-fill any NaNs (coastal SST has occasional gaps)
    df = df.ffill().bfill()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df
