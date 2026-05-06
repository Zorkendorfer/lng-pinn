"""Synthetic LNG cargo composition trajectories.

Archetypal compositions sourced from GIIGNL Annual Report 2023 and IGU World LNG Report 2023.
Each vector is (CH4, C2H6, C3H8, nC4H10, iC4H10, N2) mole fractions summing to 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Archetypal mole-fraction vectors (CH4, C2H6, C3H8, nC4, iC4, N2)
# Source: GIIGNL Annual Report 2023, Table 4 (typical delivered compositions)
ARCHETYPES: dict[str, tuple[float, ...]] = {
    "US_Gulf":   (0.906, 0.063, 0.017, 0.004, 0.004, 0.006),
    "Qatar":     (0.900, 0.060, 0.020, 0.010, 0.005, 0.005),
    "Norway":    (0.920, 0.045, 0.015, 0.008, 0.006, 0.006),
    "Algeria":   (0.870, 0.090, 0.025, 0.008, 0.005, 0.002),
    "US_East":   (0.942, 0.042, 0.010, 0.003, 0.002, 0.001),
}

CARGO_CYCLE_DAYS = 12          # average days between cargoes
BLEND_DAYS = 5                 # linear blend period after new cargo arrives


def build_composition_series(
    index: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate hourly LNG composition time series for a given time index.

    A new cargo arrives every ~CARGO_CYCLE_DAYS days drawn from ARCHETYPES.
    Transitions are linearly blended over BLEND_DAYS days.

    Returns:
        DataFrame with columns matching ARCHETYPES keys' component names,
        indexed by `index`.
    """
    rng = np.random.default_rng(seed)
    archetype_names = list(ARCHETYPES.keys())
    compositions = np.array([ARCHETYPES[k] for k in archetype_names])
    n_components = compositions.shape[1]

    hours = len(index)
    result = np.zeros((hours, n_components))

    cargo_hours = int(CARGO_CYCLE_DAYS * 24)
    blend_hours = int(BLEND_DAYS * 24)

    current_idx = rng.integers(len(archetype_names))
    next_idx = current_idx
    hours_since_cargo = 0

    for h in range(hours):
        if hours_since_cargo == 0:
            current_idx = next_idx
            next_idx = rng.integers(len(archetype_names))

        if hours_since_cargo < blend_hours:
            alpha = hours_since_cargo / blend_hours
            x = (1 - alpha) * compositions[current_idx] + alpha * compositions[next_idx]
        else:
            x = compositions[next_idx]

        result[h] = x
        hours_since_cargo = (hours_since_cargo + 1) % cargo_hours

    columns = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]
    df = pd.DataFrame(result, index=index, columns=columns)
    # Normalise rows to sum to 1 (guard against floating-point drift)
    df = df.div(df.sum(axis=1), axis=0)
    return df
