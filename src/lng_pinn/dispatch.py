"""Economic dispatch optimizer using PINN surrogate as cost model.

Approach: Path A (plan §4) — discretise send-out flow into levels,
evaluate PINN cost per level per hour in preprocessing, solve MILP with SciPy/HiGHS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from lng_pinn.pinn import PINNMLP, Scaler

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


def _pinn_cost_table(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    flow_levels: np.ndarray,
) -> np.ndarray:
    """Pre-compute cost (EUR/h) for each (hour, flow_level) pair.

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

    with torch.no_grad():
        X = torch.from_numpy(X_np)
        W_total = scaler.unscale_y(model(scaler.scale_x(X)))[:, 1].numpy()  # (T*L,)

    W_total = W_total.reshape(T, L)  # (T, L)
    price = horizon_df["price_eur_mwh"].values[:, None]  # (T, 1)
    # cost (EUR/h) = price (EUR/MWh) * W_total (kWh/kg) * m_dot (kg/s) * 3600 s / 1000
    cost_table = price * W_total * flow_levels[None, :] * 3600.0 / 1000.0
    return cost_table.astype(np.float64)


def optimize(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    inv0: float = 0.5,
) -> Schedule:
    """Run dispatch optimisation over horizon_df.

    Args:
        horizon_df: DataFrame with hourly rows; columns: price_eur_mwh, T_amb, T_sw,
                    CH4, C2H6, C3H8, nC4H10, iC4H10, N2.
        model:      Trained PINN.
        scaler:     Matching Scaler.
        demand_kg:  Total send-out requirement (kg) over the horizon.
        inv0:       Initial tank level (fraction of TANK_CAP).
    """
    T = len(horizon_df)
    flow_levels = np.linspace(M_DOT_MIN, M_DOT_MAX, N_FLOW_LEVELS)
    cost_table = _pinn_cost_table(horizon_df, model, scaler, flow_levels)

    if not np.isfinite(cost_table).all():
        raise ValueError("PINN produced non-finite dispatch costs")

    n_vars = T * N_FLOW_LEVELS
    flow_kg_h = flow_levels * 3600.0

    # --- one-level-per-hour + demand (T+1 rows) ---
    # Build as dense then convert; T+1 rows is tiny.
    eq_block = np.zeros((T + 1, n_vars), dtype=np.float64)
    for t in range(T):
        eq_block[t, t * N_FLOW_LEVELS : (t + 1) * N_FLOW_LEVELS] = 1.0
    for t in range(T):
        eq_block[T, t * N_FLOW_LEVELS : (t + 1) * N_FLOW_LEVELS] = flow_kg_h
    lb = np.full(T + 1, -np.inf)
    ub = np.full(T + 1, np.inf)
    lb[:T] = 1.0
    ub[:T] = 1.0
    lb[T] = demand_kg

    # Inventory bounds: cumulative outflow after each hour must stay in [min, max].
    # Build a (T, T*L) lower-triangular block matrix in CSR directly.
    # Row t: sum of flow_kg_h[l] * x[prev_t, l] for prev_t <= t, all l.
    min_cumulative = (inv0 - TANK_MAX) * TANK_CAP
    max_cumulative = (inv0 - TANK_MIN) * TANK_CAP
    # Lower-triangular cumulative block: entry (t, prev_t*L+l) = flow_kg_h[l] for prev_t<=t.
    # Build via Kronecker: tril(ones(T,T)) ⊗ flow_kg_h row.
    tril_mask = np.tril(np.ones((T, T), dtype=np.float64))  # (T, T)
    # Each "block column" prev_t has L sub-columns with values flow_kg_h.
    # Reshape to (T, T*L): repeat each tril column L times, scale by flow_kg_h.
    inv_dense = np.kron(tril_mask, flow_kg_h[None, :])  # (T, T*L)
    inv_block = csr_matrix(inv_dense)
    constraint_csr = vstack([csr_matrix(eq_block), inv_block])
    lb = np.concatenate([lb, np.full(T, min_cumulative)])
    ub = np.concatenate([ub, np.full(T, max_cumulative)])

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

    return Schedule(
        m_dot=m_dot_out,
        cost_eur=cost_out,
        tank_level=inv_out,
        total_cost=float(cost_out.sum()),
    )
