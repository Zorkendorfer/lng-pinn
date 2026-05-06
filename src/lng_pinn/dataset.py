"""Build training dataset and hourly timeseries from the CoolProp plant simulator."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from lng_pinn.composition import build_composition_series
from lng_pinn.market import load_da_prices, pull_weather

PROCESSED_DIR = Path("data/processed")

# Operating envelope bounds  [min, max]
BOUNDS = {
    "CH4": (0.82, 0.96),
    "C2H6": (0.02, 0.12),
    "C3H8": (0.005, 0.035),
    "nC4H10": (0.001, 0.015),
    "iC4H10": (0.001, 0.010),
    # N2 is computed as remainder
    "m_dot": (10.0, 80.0),  # kg/s
    "T_amb": (258.0, 308.0),  # K
    "T_sw": (271.0, 298.0),  # K
}


def _sample_compositions(lhs_cols: np.ndarray) -> np.ndarray:
    """Map first 5 LHS columns to valid mole fractions (sum-to-1 constrained)."""
    raw = lhs_cols.copy()
    ranges = [(0.82, 0.96), (0.02, 0.12), (0.005, 0.035), (0.001, 0.015), (0.001, 0.010)]
    for i, (lo, hi) in enumerate(ranges):
        raw[:, i] = lo + raw[:, i] * (hi - lo)
    n2 = np.clip(1.0 - raw.sum(axis=1), 0.0, 0.02)
    total = raw.sum(axis=1) + n2
    raw = raw / total[:, None]
    n2 = n2 / total
    return np.column_stack([raw, n2])


def _simulate_one(args: tuple[Any, ...]) -> dict[str, float]:
    """Top-level function so it can be pickled by ProcessPoolExecutor."""
    from lng_pinn.plant import simulate  # imported inside worker to avoid pickling issues

    x, m_dot, T_amb, T_sw = args
    out = simulate(x, float(m_dot), float(T_amb), float(T_sw))
    return {
        "CH4": x[0],
        "C2H6": x[1],
        "C3H8": x[2],
        "nC4H10": x[3],
        "iC4H10": x[4],
        "N2": x[5],
        "m_dot": float(m_dot),
        "T_amb": float(T_amb),
        "T_sw": float(T_sw),
        "W_pump": out.W_pump,
        "W_trim": out.W_trim,
        "W_total": out.W_total,
        "T_out": out.T_out,
        "Q_sw": out.Q_sw,
        "exergy_destruction": out.exergy_destruction,
    }


def build_training_set(
    N: int = 20_000,
    seed: int = 0,
    workers: int | None = None,
) -> pd.DataFrame:
    """Sample N operating points via LHS, simulate in parallel, return DataFrame.

    Columns: CH4, C2H6, C3H8, nC4H10, iC4H10, N2, m_dot, T_amb, T_sw,
             W_pump, W_trim, W_total, T_out, Q_sw, exergy_destruction
    """
    sampler = LatinHypercube(d=8, seed=seed)
    samples = sampler.random(N)

    compositions = _sample_compositions(samples[:, :5])
    m_dot = BOUNDS["m_dot"][0] + samples[:, 5] * (BOUNDS["m_dot"][1] - BOUNDS["m_dot"][0])
    T_amb = BOUNDS["T_amb"][0] + samples[:, 6] * (BOUNDS["T_amb"][1] - BOUNDS["T_amb"][0])
    T_sw = BOUNDS["T_sw"][0] + samples[:, 7] * (BOUNDS["T_sw"][1] - BOUNDS["T_sw"][0])

    args = [(tuple(compositions[i]), m_dot[i], T_amb[i], T_sw[i]) for i in range(N)]

    n_workers = workers or max(1, (os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        records = list(executor.map(_simulate_one, args, chunksize=50))

    df = pd.DataFrame(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    return df


def build_timeseries(
    start: str = "2021-01-01",
    end: str = "2024-01-01",
    zone: str = "LT",
    seed: int = 42,
) -> pd.DataFrame:
    """Build hourly timeseries of (price, T_amb, T_sw, composition) for dispatch.

    Joins ENTSO-E day-ahead prices, Open-Meteo weather, and synthetic
    cargo composition trajectories on a common UTC hourly index.

    Returns:
        DataFrame saved to data/processed/timeseries.parquet with columns:
        price_eur_mwh, T_amb (K), T_sw (K), CH4, C2H6, C3H8, nC4H10, iC4H10, N2.
    """
    prices = load_da_prices(start, end, zone=zone)
    weather = pull_weather(start, end)

    # Align weather to price index (resample/reindex to hourly UTC)
    idx = prices.index
    weather = weather.reindex(idx, method="nearest", tolerance="1h")

    # Synthetic composition on same index
    comp = build_composition_series(idx, seed=seed)

    ts = pd.concat([prices, weather, comp], axis=1)
    ts = ts.dropna(subset=["price_eur_mwh"])  # drop any residual gaps

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(PROCESSED_DIR / "timeseries.parquet")
    return ts
