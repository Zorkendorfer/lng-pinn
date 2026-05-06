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
from scipy.sparse import lil_matrix

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
    """
    T = len(horizon_df)
    L = len(flow_levels)
    cost_table = np.zeros((T, L))

    comp_cols = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]

    with torch.no_grad():
        for l_idx, m in enumerate(flow_levels):
            rows = []
            for _, row in horizon_df.iterrows():
                x = [row[c] for c in comp_cols] + [m, row["T_amb"], row["T_sw"]]
                rows.append(x)
            X = torch.tensor(rows, dtype=torch.float32)
            X_norm = scaler.scale_x(X)
            y_norm = model(X_norm)
            y = scaler.unscale_y(y_norm).numpy()
            W_total = y[:, 1]  # kWh/kg
            price = horizon_df["price_eur_mwh"].values
            cost_table[:, l_idx] = price * W_total * m * 3600.0 / 1000.0  # EUR/h

    return cost_table


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

    n_constraints = T + 1 + T
    constraint = lil_matrix((n_constraints, n_vars), dtype=float)
    lb = np.full(n_constraints, -np.inf)
    ub = np.full(n_constraints, np.inf)

    # One level per hour.
    for t in range(T):
        row = t
        start = t * N_FLOW_LEVELS
        constraint[row, start : start + N_FLOW_LEVELS] = 1.0
        lb[row] = 1.0
        ub[row] = 1.0

    # Demand over the full horizon.
    demand_row = T
    for t in range(T):
        start = t * N_FLOW_LEVELS
        constraint[demand_row, start : start + N_FLOW_LEVELS] = flow_kg_h
    lb[demand_row] = demand_kg

    # Inventory bounds after each hour, with inv[t] = inv0 - cumulative_flow / TANK_CAP.
    min_cumulative = (inv0 - TANK_MAX) * TANK_CAP
    max_cumulative = (inv0 - TANK_MIN) * TANK_CAP
    for t in range(T):
        row = T + 1 + t
        for prev_t in range(t + 1):
            start = prev_t * N_FLOW_LEVELS
            constraint[row, start : start + N_FLOW_LEVELS] = flow_kg_h
        lb[row] = min_cumulative
        ub[row] = max_cumulative

    result = milp(
        c=cost_table.reshape(n_vars),
        integrality=np.ones(n_vars),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(constraint.tocsr(), lb, ub),
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
