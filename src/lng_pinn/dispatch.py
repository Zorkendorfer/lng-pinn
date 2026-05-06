"""Economic dispatch optimizer using PINN surrogate as cost model.

Approach: Path A (plan §4) — discretise send-out flow into levels,
evaluate PINN cost per level per hour in preprocessing, solve MILP with Pyomo + CBC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch

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

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.L = pyo.RangeSet(0, N_FLOW_LEVELS - 1)

    m.x = pyo.Var(m.T, m.L, domain=pyo.Binary)
    m.inv = pyo.Var(pyo.RangeSet(0, T), domain=pyo.NonNegativeReals, bounds=(TANK_MIN, TANK_MAX))

    # One level per hour
    m.one_level = pyo.Constraint(m.T, rule=lambda m, t: sum(m.x[t, lv] for lv in m.L) == 1)

    # Inventory dynamics (dt = 1 h = 3600 s)
    def inv_dynamics(m: pyo.ConcreteModel, t: int) -> pyo.Expression:
        flow_kg_h = sum(m.x[t, lv] * flow_levels[lv] * 3600.0 for lv in m.L)
        return m.inv[t + 1] == m.inv[t] - flow_kg_h / TANK_CAP

    m.inv_dyn = pyo.Constraint(m.T, rule=inv_dynamics)
    m.inv[0].fix(inv0)

    # Demand constraint
    m.demand = pyo.Constraint(
        expr=sum(sum(m.x[t, lv] * flow_levels[lv] * 3600.0 for lv in m.L) for t in m.T) >= demand_kg
    )

    # Objective
    m.obj = pyo.Objective(
        expr=sum(cost_table[t, lv] * m.x[t, lv] for t in m.T for lv in m.L),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("cbc")
    solver.solve(m, tee=False)

    m_dot_out = np.array([sum(pyo.value(m.x[t, lv]) * flow_levels[lv] for lv in m.L) for t in m.T])
    cost_out = np.array([sum(cost_table[t, lv] * pyo.value(m.x[t, lv]) for lv in m.L) for t in m.T])
    inv_out = np.array([pyo.value(m.inv[t]) for t in range(T + 1)])

    return Schedule(
        m_dot=m_dot_out,
        cost_eur=cost_out,
        tank_level=inv_out,
        total_cost=float(cost_out.sum()),
    )
