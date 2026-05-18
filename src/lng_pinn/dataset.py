"""Build training dataset and hourly timeseries from the CoolProp plant simulator."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

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


def _simulate_one(args: tuple[Any, ...]) -> dict[str, float] | None:
    """Top-level function so it can be pickled by ProcessPoolExecutor."""
    import CoolProp.CoolProp as CP

    from lng_pinn.plant import (  # imported inside worker to avoid pickling issues
        J_TO_KWH,
        P_IN,
        P_OUT_DEFAULT,
        T_IN,
        T_SENDOUT,
        pump_efficiency,
        simulate,
    )
    from lng_pinn.thermo import get_state

    x, m_dot, T_amb, T_sw = args
    try:
        out = simulate(x, float(m_dot), float(T_amb), float(T_sw))
    except ValueError:
        # CoolProp can't find a density solution (two-phase or near-critical region)
        return None

    # Enthalpy at storage and send-out conditions (needed for PINN energy balance loss)
    state = get_state(x)
    state.update(CP.PT_INPUTS, P_IN, T_IN)
    h_in_per_kg = state.hmolar() / state.molar_mass()   # J/kg
    rho_in = state.rhomass()                             # kg/m^3
    state.update(CP.PT_INPUTS, P_OUT_DEFAULT, T_SENDOUT)
    h_out_per_kg = state.hmolar() / state.molar_mass()  # J/kg

    # Analytical pump work (incompressible liquid, flow-dependent efficiency)
    eta = pump_efficiency(float(m_dot))
    W_pump_expected = (P_OUT_DEFAULT - P_IN) / rho_in / eta * J_TO_KWH  # kWh/kg

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
        "h_in_per_kg": h_in_per_kg,
        "h_out_per_kg": h_out_per_kg,
        "W_pump_expected": W_pump_expected,
    }


def build_training_set(
    N: int = 20_000,
    seed: int = 0,
    workers: int | None = None,
) -> pd.DataFrame:
    """Sample N operating points via LHS, simulate in parallel, return DataFrame.

    Columns: CH4, C2H6, C3H8, nC4H10, iC4H10, N2, m_dot, T_amb, T_sw,
             W_pump, W_trim, W_total, T_out, Q_sw, exergy_destruction,
             h_in_per_kg, h_out_per_kg, W_pump_expected
    """
    sampler = LatinHypercube(d=8, seed=seed)
    samples = sampler.random(N)

    compositions = _sample_compositions(samples[:, :5])
    m_dot = BOUNDS["m_dot"][0] + samples[:, 5] * (BOUNDS["m_dot"][1] - BOUNDS["m_dot"][0])
    T_amb = BOUNDS["T_amb"][0] + samples[:, 6] * (BOUNDS["T_amb"][1] - BOUNDS["T_amb"][0])
    T_sw = BOUNDS["T_sw"][0] + samples[:, 7] * (BOUNDS["T_sw"][1] - BOUNDS["T_sw"][0])

    args = [(tuple(compositions[i]), m_dot[i], T_amb[i], T_sw[i]) for i in range(N)]

    n_workers = workers or max(1, min(4, os.cpu_count() or 1))
    if n_workers == 1:
        raw = [_simulate_one(arg) for arg in tqdm(args, total=N, desc="Simulating", unit="pts")]
    else:
        raw = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_simulate_one, arg) for arg in args]
            for future in tqdm(as_completed(futures), total=N, desc="Simulating", unit="pts"):
                raw.append(future.result())

    records = [r for r in raw if r is not None]
    n_skipped = N - len(records)
    if n_skipped:
        import warnings
        warnings.warn(
            f"Skipped {n_skipped}/{N} ({100*n_skipped/N:.1f}%) points where CoolProp "
            "found no density solution (two-phase or near-critical region).",
            stacklevel=2,
        )

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
