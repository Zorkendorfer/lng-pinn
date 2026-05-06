"""Composition-blind dispatch baseline.

Uses the annual mean composition instead of the actual hourly composition.
Everything else (prices, temperatures, demand, inventory) is identical to
the composition-aware dispatch.
"""

from __future__ import annotations

import pandas as pd

from lng_pinn.dispatch import Schedule, optimize
from lng_pinn.pinn import PINNMLP, Scaler

COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]


def optimize_blind(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    inv0: float = 0.5,
) -> Schedule:
    """Run dispatch with composition fixed to the horizon mean."""
    blind_df = horizon_df.copy()
    mean_comp = horizon_df[COMP_COLS].mean()
    # Normalise so fractions sum to 1
    mean_comp = mean_comp / mean_comp.sum()
    for col in COMP_COLS:
        blind_df[col] = mean_comp[col]

    return optimize(blind_df, model, scaler, demand_kg, inv0)
