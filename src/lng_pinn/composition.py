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


COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]


def build_composition_series_from_csv(
    index: pd.DatetimeIndex,
    csv_path: str,
    blend_days: float = BLEND_DAYS,
) -> pd.DataFrame:
    """Hourly composition series from an *exogenous* cargo-arrival schedule.

    Rework plan item 6: lets the backtest run on real or semi-real cargo
    arrivals and assayed compositions instead of the synthetic GIIGNL schedule,
    so the headline cannot be dismissed as an artefact of the synthetic process.

    The CSV must have an ``arrival`` datetime column plus the six mole-fraction
    columns (``CH4, C2H6, C3H8, nC4H10, iC4H10, N2``). Rows are sorted by
    arrival; each transition is linearly blended over ``blend_days`` (the same
    tank-mixing assumption as :func:`build_composition_series`, so results are
    comparable). Cargoes are renormalised to sum to 1, and the first cargo fills
    the tank for any hours preceding the first arrival.

    To use it in a backtest, swap the composition columns of the timeseries:
    ``ts[COMP_COLS] = build_composition_series_from_csv(ts.index, path)[COMP_COLS]``.
    """
    idx = pd.DatetimeIndex(index)
    cargo = pd.read_csv(csv_path)
    cargo["arrival"] = pd.to_datetime(cargo["arrival"], utc=True)
    cargo = cargo.sort_values("arrival").reset_index(drop=True)
    if cargo.empty:
        raise ValueError(f"no cargo rows in {csv_path}")

    comps = cargo[COMP_COLS].to_numpy(dtype=float)
    comps = comps / comps.sum(axis=1, keepdims=True)
    # Use pandas arithmetic throughout: tz-aware .to_numpy() yields object-dtype
    # Timestamps, which cannot be subtracted from a datetime64 in numpy.
    arr_h = ((cargo["arrival"] - idx[0]) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    h = ((idx - idx[0]) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    blend_hours = float(blend_days * 24)

    out = np.zeros((len(idx), 6), dtype=float)
    anchor = comps[0].copy()
    for j in range(len(comps)):
        a = float(arr_h[j])
        end = float(arr_h[j + 1]) if j + 1 < len(comps) else h[-1] + 1.0
        seg = (h >= a) & (h < end)
        if seg.any():
            w = np.clip((h[seg] - a) / blend_hours, 0.0, 1.0)
            out[seg] = (1 - w)[:, None] * anchor[None, :] + w[:, None] * comps[j][None, :]
        w_end = np.clip((end - a) / blend_hours, 0.0, 1.0)
        anchor = (1 - w_end) * anchor + w_end * comps[j]
    out[h < float(arr_h[0])] = comps[0]  # hours before the first arrival

    df = pd.DataFrame(out, index=idx, columns=COMP_COLS)
    return df.div(df.sum(axis=1), axis=0)
