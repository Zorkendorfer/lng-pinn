"""Build (X, y) training dataset by sampling the CoolProp plant simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from lng_pinn.plant import simulate
from lng_pinn.thermo import SPECIES_KEYS

PROCESSED_DIR = Path("data/processed")

# Operating envelope bounds  [min, max]
BOUNDS = {
    "CH4":    (0.82, 0.96),
    "C2H6":   (0.02, 0.12),
    "C3H8":   (0.005, 0.035),
    "nC4H10": (0.001, 0.015),
    "iC4H10": (0.001, 0.010),
    # N2 is computed as remainder
    "m_dot":  (10.0, 80.0),    # kg/s
    "T_amb":  (258.0, 308.0),  # K
    "T_sw":   (271.0, 298.0),  # K
}


def _sample_compositions(lhs_cols: np.ndarray) -> np.ndarray:
    """Map first 5 LHS columns to valid mole fractions (sum-to-1 constrained)."""
    raw = lhs_cols.copy()
    # Scale each to its archetype range
    ranges = [(0.82, 0.96), (0.02, 0.12), (0.005, 0.035), (0.001, 0.015), (0.001, 0.010)]
    for i, (lo, hi) in enumerate(ranges):
        raw[:, i] = lo + raw[:, i] * (hi - lo)
    n2 = np.clip(1.0 - raw.sum(axis=1), 0.0, 0.02)
    total = raw.sum(axis=1) + n2
    raw = raw / total[:, None]
    n2 = n2 / total
    return np.column_stack([raw, n2])


def build_training_set(N: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Sample N operating points via LHS, simulate each, return DataFrame.

    Columns: CH4, C2H6, C3H8, nC4H10, iC4H10, N2, m_dot, T_amb, T_sw,
             W_pump, W_trim, W_total, T_out, Q_sw, exergy_destruction
    """
    sampler = LatinHypercube(d=8, seed=seed)
    samples = sampler.random(N)  # (N, 8) in [0,1]

    compositions = _sample_compositions(samples[:, :5])  # (N, 6)

    m_dot = BOUNDS["m_dot"][0] + samples[:, 5] * (BOUNDS["m_dot"][1] - BOUNDS["m_dot"][0])
    T_amb = BOUNDS["T_amb"][0] + samples[:, 6] * (BOUNDS["T_amb"][1] - BOUNDS["T_amb"][0])
    T_sw  = BOUNDS["T_sw"][0]  + samples[:, 7] * (BOUNDS["T_sw"][1]  - BOUNDS["T_sw"][0])

    records = []
    for i in range(N):
        x = tuple(compositions[i])
        out = simulate(x, float(m_dot[i]), float(T_amb[i]), float(T_sw[i]))
        records.append({
            "CH4": x[0], "C2H6": x[1], "C3H8": x[2],
            "nC4H10": x[3], "iC4H10": x[4], "N2": x[5],
            "m_dot": m_dot[i], "T_amb": T_amb[i], "T_sw": T_sw[i],
            "W_pump": out.W_pump, "W_trim": out.W_trim, "W_total": out.W_total,
            "T_out": out.T_out, "Q_sw": out.Q_sw,
            "exergy_destruction": out.exergy_destruction,
        })

    df = pd.DataFrame(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    return df
