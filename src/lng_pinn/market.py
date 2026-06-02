"""ENTSO-E day-ahead price ingestion and Open-Meteo weather fetching.

v1.5: weather is now site-aware. ``pull_weather`` and ``build_timeseries``
take optional ``lat``/``lon`` (or a named site via :data:`SITES`) so the
same code can be reused for any FSRU site. Cached weather files include
the lat/lon in their filename so Klaipėda and Wilhelmshaven coexist
without clobbering each other.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from entsoe import EntsoePandasClient

RAW_DIR = Path("data/raw")
ZONE = "LT"  # ENTSO-E bidding zone (default — Lithuania)

# Independence FSRU, Klaipėda — defaults preserved for back-compat.
LAT = 55.71
LON = 21.13


# Known FSRU sites: name → (lat, lon, ENTSO-E bidding zone).
# Lat/lon are approximate terminal coordinates; ENTSO-E zone is the
# spot-market bidding zone the terminal lives in.
SITES: dict[str, tuple[float, float, str]] = {
    "klaipeda":      (55.71, 21.13, "LT"),     # Independence — reference site
    "wilhelmshaven": (53.52, 8.13,  "DE_LU"),  # Höegh Esperanza / Excelsior / Excelerate
    "brunsbuttel":   (53.89, 9.13,  "DE_LU"),  # Höegh Gannet
    "stade":         (53.61, 9.47,  "DE_LU"),
    "mukran":        (54.51, 13.71, "DE_LU"),  # Energos Power (Rügen)
    "lubmin":        (54.13, 13.62, "DE_LU"),  # Neptune (shut down 2024)
}


def resolve_site(site: str) -> tuple[float, float, str]:
    """Look up a known site by name. Case-insensitive; underscores ignored.

    Returns ``(lat, lon, zone)``. Raises ``KeyError`` with an enumerated
    list of valid names if the site is unknown.
    """
    key = site.lower().replace("_", "").replace(" ", "")
    aliases = {k.replace("_", "").replace(" ", ""): k for k in SITES}
    if key not in aliases:
        valid = ", ".join(sorted(SITES))
        raise KeyError(f"Unknown site {site!r}. Valid: {valid}")
    return SITES[aliases[key]]


def _token() -> str:
    token = os.environ.get("ENTSOE_API_TOKEN", "")
    if not token:
        raise RuntimeError("Set ENTSOE_API_TOKEN in your .env file.")
    return token


def pull_da_prices(start: str, end: str, zone: str = ZONE) -> pd.DataFrame:
    """Pull day-ahead prices from ENTSO-E and cache to parquet (year-by-year).

    Iterates over calendar years so each yearly file matches what load_da_prices
    expects. Already-cached years are skipped.

    Args:
        start: ISO date string, e.g. "2021-01-01".
        end:   ISO date string, e.g. "2026-01-01".
        zone:  ENTSO-E bidding zone code.

    Returns:
        DataFrame with DatetimeTZDtype index (UTC) and column "price_eur_mwh".
    """
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    client: EntsoePandasClient | None = None
    frames = []
    for year in range(start_year, end_year):
        y_start = f"{year}-01-01"
        y_end = f"{year + 1}-01-01"
        cache_path = RAW_DIR / f"da_prices_{zone}_{y_start}_{y_end}.parquet"
        if cache_path.exists():
            frames.append(pd.read_parquet(cache_path))
            continue

        if client is None:
            client = EntsoePandasClient(api_key=_token())

        series = client.query_day_ahead_prices(
            zone,
            start=pd.Timestamp(y_start, tz="UTC"),
            end=pd.Timestamp(y_end, tz="UTC"),
        )
        df = series.to_frame(name="price_eur_mwh")
        df.index.name = "utc_time"
        df.to_parquet(cache_path)
        frames.append(df)

    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    return result


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


def pull_weather(
    start: str,
    end: str,
    lat: float = LAT,
    lon: float = LON,
) -> pd.DataFrame:
    """Fetch hourly T_amb and T_sw at (``lat``, ``lon``) from Open-Meteo.

    Uses ERA5 reanalysis (archive API) for air temperature and the Marine
    API for sea surface temperature. The cache filename embeds the
    coordinates so different sites coexist on disk:

        data/raw/weather_<lat>_<lon>_<start>_<end>.parquet

    A legacy file ``weather_<start>_<end>.parquet`` (without coordinates)
    is recognised only when ``lat``/``lon`` equal the module defaults
    (Klaipėda) — that keeps the v1.4 Lithuanian cache valid without a
    re-pull.

    Args:
        start, end: ISO date strings (UTC).
        lat, lon:   site coordinates. Default: Klaipėda.

    Returns:
        DataFrame indexed by UTC hour with columns T_amb (K), T_sw (K).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"weather_{lat:.2f}_{lon:.2f}_{start}_{end}.parquet"

    # Legacy filename support (no lat/lon in cache name) — Klaipėda only.
    legacy_path = RAW_DIR / f"weather_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    if legacy_path.exists() and (lat, lon) == (LAT, LON):
        return pd.read_parquet(legacy_path)

    # Open-Meteo end date is inclusive; subtract one day
    end_date = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Air temperature (ERA5 reanalysis)
    r_air = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
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
            "latitude": lat,
            "longitude": lon,
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

    df.to_parquet(cache_path)
    return df
