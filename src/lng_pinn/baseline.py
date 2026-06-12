"""Dispatch baselines for composition-aware backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lng_pinn.dispatch import TANK_CAP, Schedule, optimize
from lng_pinn.pinn import PINNMLP, Scaler, build_aux
from lng_pinn.thermo import co2_per_kg_fuel

COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]


def _with_fixed_composition(
    horizon_df: pd.DataFrame,
    composition: pd.Series,
) -> pd.DataFrame:
    blind_df = horizon_df.copy()
    composition = composition / composition.sum()
    for col in COMP_COLS:
        blind_df[col] = composition[col]
    return blind_df


def optimize_blind_horizon(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    inv0: float = 0.5,
    carbon_price_eur_per_t: float = 0.0,
    demand_ub_kg: float | None = None,
) -> Schedule:
    """Run dispatch with composition fixed to the horizon mean."""
    blind_df = _with_fixed_composition(horizon_df, horizon_df[COMP_COLS].mean())
    return optimize(
        blind_df, model, scaler, demand_kg, inv0, carbon_price_eur_per_t,
        demand_ub_kg=demand_ub_kg,
    )


def optimize_blind_annual(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    annual_composition: pd.Series,
    inv0: float = 0.5,
    carbon_price_eur_per_t: float = 0.0,
) -> Schedule:
    """Run dispatch with composition fixed to the full-backtest annual mean."""
    blind_df = _with_fixed_composition(horizon_df, annual_composition)
    return optimize(blind_df, model, scaler, demand_kg, inv0, carbon_price_eur_per_t)


def optimize_constant_flow(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    inv0: float = 0.5,
    carbon_price_eur_per_t: float = 0.0,
) -> Schedule:
    """Dispatch at the constant flow required to meet horizon demand."""
    m_dot = demand_kg / (len(horizon_df) * 3600.0)
    rows = []
    for _, row in horizon_df.iterrows():
        rows.append([row[c] for c in COMP_COLS] + [m_dot, row["T_amb"], row["T_sw"]])

    with torch.no_grad():
        X = torch.tensor(rows, dtype=torch.float32)
        aux = build_aux(X[:, :6].numpy(), X[:, 6].numpy())
        y = scaler.unscale_y(model(scaler.scale_x(X), aux, scaler=scaler)).numpy()

    W_total = y[:, 1]
    price = horizon_df["price_eur_mwh"].values
    m_dot_out = pd.Series(m_dot, index=horizon_df.index).to_numpy()
    cost_out = price * W_total * m_dot * 3600.0 / 1000.0
    if carbon_price_eur_per_t > 0.0:
        co2_factor = np.array(
            [co2_per_kg_fuel(tuple(float(v) for v in row)) for row in horizon_df[COMP_COLS].values],
            dtype=np.float64,
        )
        cost_out = cost_out + carbon_price_eur_per_t * co2_factor * m_dot * 3.6
    cumulative_flow = pd.Series(m_dot_out * 3600.0).cumsum().to_numpy()
    tank_level = inv0 - pd.Series([0.0, *cumulative_flow]).to_numpy() / TANK_CAP
    return Schedule(
        m_dot=m_dot_out,
        cost_eur=cost_out,
        tank_level=tank_level,
        total_cost=float(cost_out.sum()),
    )


def optimize_blind_lagged(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    lagged_composition: pd.Series,
    inv0: float = 0.5,
    carbon_price_eur_per_t: float = 0.0,
    demand_ub_kg: float | None = None,
) -> Schedule:
    """Run dispatch with composition fixed to the value at the start of the window.

    Models the realistic operator assumption: current measured composition is known
    and held constant for the horizon, without knowing how it will evolve.
    """
    blind_df = _with_fixed_composition(horizon_df, lagged_composition)
    return optimize(
        blind_df, model, scaler, demand_kg, inv0, carbon_price_eur_per_t,
        demand_ub_kg=demand_ub_kg,
    )


def optimize_blind(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    inv0: float = 0.5,
    carbon_price_eur_per_t: float = 0.0,
) -> Schedule:
    """Backward-compatible alias for the horizon-mean blind baseline."""
    return optimize_blind_horizon(
        horizon_df, model, scaler, demand_kg, inv0, carbon_price_eur_per_t
    )


def optimize_perfect_foresight_block(
    block_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_per_hour_kg: float,
    inv0: float,
    carbon_price_eur_per_t: float = 0.0,
) -> Schedule:
    """v1.4 A — perfect-foresight dispatch over one block with true composition.

    The full-horizon single-shot LP is intractable (the inventory constraint is
    O(T²)), so the oracle runs over non-overlapping blocks whose inventory is
    carried across by the caller, with cargo injected *between* blocks exactly
    as the rolling drivers inject it between windows. When the block length is a
    multiple of the cargo cycle and aligned to it, there are no mid-block cargo
    arrivals, so this is a plain ``optimize`` call over the block (the tested
    path) — no ``cargo_frac_cumulative`` shimming required.

    Within a block it sees the entire block's true hourly composition and price
    at once — no rolling-window myopia, no composition blinding — so for blocks
    longer than the rolling strategies' 7-day lookahead it is an
    extended/perfect-foresight reference. In practice

        saving(oracle) >= saving(aware) >= saving(lagged)

    at every carbon price; validate this ordering on the first run (a violation
    means the block is shorter than the rolling lookahead, or a dispatch bug).
    """
    demand_kg = demand_per_hour_kg * len(block_df)
    return optimize(
        block_df, model, scaler, demand_kg, inv0,
        carbon_price_eur_per_t=carbon_price_eur_per_t,
    )
