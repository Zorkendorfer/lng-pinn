"""Economic dispatch optimizer using PINN surrogate as cost model.

Approach: Path A (plan §4) — discretise send-out flow into levels,
evaluate PINN cost per level per hour in preprocessing, solve MILP with SciPy/HiGHS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from lng_pinn.pinn import PINNMLP, Scaler, build_aux
from lng_pinn.thermo import co2_per_kg_fuel

N_FLOW_LEVELS = 15  # discretisation resolution
M_DOT_MIN = 10.0  # kg/s — minimum stable turndown
M_DOT_MAX = 80.0  # kg/s — maximum send-out
TANK_MIN = 0.05  # fraction of capacity
TANK_MAX = 0.95
TANK_CAP = 180_000_000.0  # kg — LNG storage capacity (Independence FSRU ~170,000 m³ × ~450 kg/m³)


@dataclass
class Schedule:
    m_dot: np.ndarray  # kg/s, shape (T,)
    cost_eur: np.ndarray  # EUR, shape (T,)
    tank_level: np.ndarray  # fraction, shape (T+1,)
    total_cost: float


COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]

# Cache of the (constant) constraint matrix per horizon length T. The equality
# block (one level per hour + total-demand row) and the lower-triangular
# inventory block depend only on T and the fixed flow-level grid, not on prices,
# composition, inventory or carbon — so for a rolling backtest with a fixed
# window length they are identical across all ~5k calls. Building them
# (a dense T+1 x T*L block + an np.kron T x T*L block + CSR conversion) on every
# call was pure waste; we build once per T and reuse. Only the bounds vary.
_CONSTRAINT_CACHE: dict[int, Any] = {}


def _constraint_matrix(T: int) -> Any:
    """Return the cached sparse constraint matrix for horizon length ``T``."""
    cached = _CONSTRAINT_CACHE.get(T)
    if cached is not None:
        return cached
    n_vars = T * N_FLOW_LEVELS
    flow_levels = np.linspace(M_DOT_MIN, M_DOT_MAX, N_FLOW_LEVELS)
    flow_kg_h = flow_levels * 3600.0
    eq_block = np.zeros((T + 1, n_vars), dtype=np.float64)
    for t in range(T):
        eq_block[t, t * N_FLOW_LEVELS : (t + 1) * N_FLOW_LEVELS] = 1.0
    for t in range(T):
        eq_block[T, t * N_FLOW_LEVELS : (t + 1) * N_FLOW_LEVELS] = flow_kg_h
    tril_mask = np.tril(np.ones((T, T), dtype=np.float64))
    inv_dense = np.kron(tril_mask, flow_kg_h[None, :])  # (T, T*L)
    constraint_csr = vstack([csr_matrix(eq_block), csr_matrix(inv_dense)])
    _CONSTRAINT_CACHE[T] = constraint_csr
    return constraint_csr


def _pinn_cost_table(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    flow_levels: np.ndarray,
    carbon_price_eur_per_t: float = 0.0,
) -> np.ndarray:
    """Pre-compute total cost (EUR/h) for each (hour, flow_level) pair.

    cost[t, l] = price[t] * W_total[t, l] * flow_levels[l] * 3.6
               + carbon_price * co2_factor[t] * flow_levels[l] * 3.6

    Returns array of shape (T, N_FLOW_LEVELS).
    Batches all (T × L) inputs in a single PINN forward pass.
    """
    T = len(horizon_df)
    L = len(flow_levels)
    comp_cols = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]

    # Base features repeated for each flow level: (T, 8)
    base = horizon_df[comp_cols + ["T_amb", "T_sw"]].values.astype(np.float32)  # (T, 8)
    # Tile: (T*L, 9) — repeat each hour L times, insert m_dot column
    base_rep = np.repeat(base, L, axis=0)  # (T*L, 8)
    m_rep = np.tile(flow_levels.astype(np.float32), T)  # (T*L,)
    X_np = np.concatenate(
        [base_rep[:, :6], m_rep[:, None], base_rep[:, 6:]], axis=1
    )  # (T*L, 9): comp + m_dot + T_amb + T_sw

    # Build aux (h_in, h_out, W_pump_expected) for all (T*L) points.
    # composition_aux is cached per unique composition so this is cheap even
    # when L = 15 flow levels are tiled across the same composition.
    aux = build_aux(X_np[:, :6], X_np[:, 6])  # (T*L, 3)

    with torch.no_grad():
        X = torch.from_numpy(X_np)
        y_pred = scaler.unscale_y(model(scaler.scale_x(X), aux, scaler=scaler))
        W_total = y_pred[:, 1].numpy()  # (T*L,)

    W_total = W_total.reshape(T, L)  # (T, L)
    price = horizon_df["price_eur_mwh"].values[:, None]  # (T, 1)
    # electricity_cost (EUR/h) = price (EUR/MWh) * W_total (kWh/kg)
    #                          * m_dot (kg/s) * (3600 s/h / 1000 kWh/MWh)
    electricity_cost = price * W_total * flow_levels[None, :] * 3600.0 / 1000.0

    if carbon_price_eur_per_t > 0.0:
        # co2_factor (kg CO2 / kg fuel) varies per hour as composition changes.
        co2_factor = np.array(
            [co2_per_kg_fuel(tuple(float(v) for v in row)) for row in horizon_df[COMP_COLS].values],
            dtype=np.float64,
        )[:, None]
        # carbon_cost (EUR/h) = price_co2 (EUR/t) * co2_factor (kg/kg) * m_dot (kg/s) * 3.6
        # (3.6 = 3600 s/h / 1000 kg/t).
        carbon_cost = carbon_price_eur_per_t * co2_factor * flow_levels[None, :] * 3.6
        return (electricity_cost + carbon_cost).astype(np.float64)
    return electricity_cost.astype(np.float64)


def optimize(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    inv0: float = 0.5,
    carbon_price_eur_per_t: float = 0.0,
    cargo_frac_cumulative: np.ndarray | None = None,
    demand_ub_kg: float | None = None,
) -> Schedule:
    """Run dispatch optimisation over horizon_df.

    Args:
        horizon_df: DataFrame with hourly rows; columns: price_eur_mwh, T_amb, T_sw,
                    CH4, C2H6, C3H8, nC4H10, iC4H10, N2.
        model:      Trained PINN.
        scaler:     Matching Scaler.
        demand_kg:  Total send-out requirement (kg) over the horizon.
        inv0:       Initial tank level (fraction of TANK_CAP).
        carbon_price_eur_per_t:
            v1.3 B1 carbon-cost term, in EUR per tonne CO2. With the default 0.0
            the objective is electricity-only and results are bit-identical to v1.2.
        cargo_frac_cumulative:
            v1.4 A — optional length-T array giving the *cumulative* cargo
            delivered (as a fraction of TANK_CAP) by the end of each hour.
            Used by the perfect-foresight oracle to model mid-horizon cargo
            arrivals inside a single long-horizon optimisation. With the
            default None there are no mid-horizon arrivals and behaviour is
            bit-identical to v1.3 (the rolling drivers inject cargo between
            windows instead).
        demand_ub_kg:
            Optional upper bound on total send-out over the horizon (kg).
            Used by the volume-matched backtest mode to pin delivered volume
            to a narrow band around the contract shortfall instead of the
            default floor-only constraint. Keep it a band (e.g. 1.001x the
            floor) rather than strict equality: the one-hot LP relaxation may
            mix two adjacent flow levels in one hour to land inside the band,
            which is operationally a flow between levels. With the default
            None, behaviour is bit-identical to the floor-only model.
    """
    T = len(horizon_df)
    flow_levels = np.linspace(M_DOT_MIN, M_DOT_MAX, N_FLOW_LEVELS)
    cost_table = _pinn_cost_table(
        horizon_df, model, scaler, flow_levels,
        carbon_price_eur_per_t=carbon_price_eur_per_t,
    )

    if not np.isfinite(cost_table).all():
        raise ValueError("PINN produced non-finite dispatch costs")

    n_vars = T * N_FLOW_LEVELS

    # Constant structure (one-hot-per-hour rows, total-demand row, lower-triangular
    # inventory block) is cached per T; only the bounds below vary per call.
    constraint_csr = _constraint_matrix(T)

    lb = np.full(T + 1, -np.inf)
    ub = np.full(T + 1, np.inf)
    lb[:T] = 1.0
    ub[:T] = 1.0
    lb[T] = demand_kg
    if demand_ub_kg is not None:
        if demand_ub_kg < demand_kg:
            raise ValueError(
                f"demand_ub_kg ({demand_ub_kg}) must be >= demand_kg ({demand_kg})"
            )
        ub[T] = demand_ub_kg

    # Inventory bounds: tank level after each hour must stay in [TANK_MIN, TANK_MAX].
    #   level(t) = inv0 + cargo_frac_cumulative(t) - cumOut(t)/TANK_CAP
    # Rearranged into cumulative-outflow bounds, the cargo term shifts the
    # per-row [min, max] window up by cargo_frac_cumulative(t)*TANK_CAP.
    if cargo_frac_cumulative is None:
        cargo_cum = np.zeros(T, dtype=np.float64)
    else:
        cargo_cum = np.asarray(cargo_frac_cumulative, dtype=np.float64)
        if cargo_cum.shape != (T,):
            raise ValueError(
                f"cargo_frac_cumulative must have shape ({T},), got {cargo_cum.shape}"
            )
    min_cumulative = (inv0 + cargo_cum - TANK_MAX) * TANK_CAP  # (T,)
    max_cumulative = (inv0 + cargo_cum - TANK_MIN) * TANK_CAP  # (T,)
    lb = np.concatenate([lb, min_cumulative])
    ub = np.concatenate([ub, max_cumulative])

    result = milp(
        c=cost_table.reshape(n_vars),
        integrality=np.zeros(n_vars),  # LP relaxation — exact due to one-hot TU structure
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(constraint_csr, lb, ub),
    )
    if not result.success:
        raise RuntimeError(f"Dispatch optimization failed: {result.message}")

    x = result.x.reshape(T, N_FLOW_LEVELS)
    m_dot_out = x @ flow_levels
    cost_out = (cost_table * x).sum(axis=1)
    cumulative_flow = np.concatenate(([0.0], np.cumsum(m_dot_out * 3600.0)))
    inv_out = inv0 - cumulative_flow / TANK_CAP
    if cargo_frac_cumulative is not None:
        # Reflect mid-horizon cargo arrivals in the reported tank level.
        # inv_out[0] is the initial level (no cargo yet); inv_out[t+1] gets
        # the cumulative cargo delivered by the end of hour t.
        inv_out[1:] = inv_out[1:] + cargo_cum

    return Schedule(
        m_dot=m_dot_out,
        cost_eur=cost_out,
        tank_level=inv_out,
        total_cost=float(cost_out.sum()),
    )
