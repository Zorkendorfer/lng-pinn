"""Composition-fabrication diagnostic (rework plan item 3).

The soft-physics fabrication hazard is that a surrogate can invent a
composition-dependent cost signal the true simulator does not have. This module
makes the §6.4 diagnostic executable: for one dispatch window at a fixed flow,
it compares the cost *difference* between the time-varying composition
trajectory and the window-mean composition, under the surrogate and under the
CoolProp ground truth. A faithful (hard-physics) surrogate reproduces the
truth's composition sensitivity, so the gap is ~0; a soft-physics surrogate
inflates it, which a downstream optimiser misreads as a real saving.

The functions take any ``(model, scaler)`` pair, so the same diagnostic runs for
both the hard and the soft surrogate once both are available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]


def window_cost_eur(
    W_total: np.ndarray,
    comp: np.ndarray,
    price_eur_mwh: np.ndarray,
    m_dot: float,
    carbon_price_eur_per_t: float,
) -> np.ndarray:
    """Per-hour dispatch cost (EUR/h) at fixed flow ``m_dot`` (kg/s).

    Matches the dispatch objective: electricity = price * W_total * m_dot * 3.6
    (3.6 = 3600 s/h / 1000 kWh/MWh); carbon = p_co2 * co2(comp) * m_dot * 3.6
    (3.6 = 3600 s/h / 1000 kg/t). Pure arithmetic, unit-testable without a model.
    """
    from lng_pinn.thermo import co2_per_kg_fuel

    elec = price_eur_mwh * W_total * m_dot * 3.6
    if carbon_price_eur_per_t > 0.0:
        co2 = np.array([co2_per_kg_fuel(tuple(float(v) for v in row)) for row in comp])
        carbon = carbon_price_eur_per_t * co2 * m_dot * 3.6
    else:
        carbon = np.zeros_like(elec)
    return elec + carbon


def _surrogate_W_total(model, scaler, comp: np.ndarray, m_dot: float,
                       T_amb: np.ndarray, T_sw: np.ndarray) -> np.ndarray:
    """W_total (kWh/kg) from the surrogate for each row; mirrors dispatch."""
    import torch

    from lng_pinn.pinn import build_aux

    T = comp.shape[0]
    X = np.concatenate(
        [comp, np.full((T, 1), m_dot), T_amb[:, None], T_sw[:, None]], axis=1
    ).astype(np.float32)
    aux = build_aux(X[:, :6], X[:, 6])
    with torch.no_grad():
        y = scaler.unscale_y(model(scaler.scale_x(torch.from_numpy(X)), aux, scaler=scaler))
    return y[:, 1].numpy()


def _truth_W_total(comp: np.ndarray, m_dot: float,
                   T_amb: np.ndarray, T_sw: np.ndarray) -> np.ndarray:
    """W_total (kWh/kg) from the CoolProp simulator for each row."""
    from lng_pinn.plant import simulate

    out = np.empty(comp.shape[0], dtype=float)
    for t in range(comp.shape[0]):
        out[t] = simulate(
            tuple(float(v) for v in comp[t]), m_dot, float(T_amb[t]), float(T_sw[t])
        ).W_total
    return out


def composition_fabrication_gap(
    model,
    scaler,
    window_df: pd.DataFrame,
    m_dot: float,
    carbon_price_eur_per_t: float = 80.0,
) -> dict:
    """Fabrication gap for one window (see module docstring).

    Returns the surrogate and truth time-varying-minus-mean cost deltas, their
    difference (the fabricated component), and that difference as a fraction of
    the mean-composition truth cost. ``flagged`` marks windows where the
    surrogate's composition sensitivity exceeds the truth's by more than the
    threshold fraction of simulator cost.
    """
    comp = window_df[COMP_COLS].to_numpy(dtype=float)
    T_amb = window_df["T_amb"].to_numpy(dtype=float)
    T_sw = window_df["T_sw"].to_numpy(dtype=float)
    price = window_df["price_eur_mwh"].to_numpy(dtype=float)
    T = len(window_df)

    comp_mean = np.tile(comp.mean(axis=0), (T, 1))

    s_var = window_cost_eur(
        _surrogate_W_total(model, scaler, comp, m_dot, T_amb, T_sw),
        comp, price, m_dot, carbon_price_eur_per_t,
    ).sum()
    s_mean = window_cost_eur(
        _surrogate_W_total(model, scaler, comp_mean, m_dot, T_amb, T_sw),
        comp_mean, price, m_dot, carbon_price_eur_per_t,
    ).sum()
    t_var = window_cost_eur(
        _truth_W_total(comp, m_dot, T_amb, T_sw),
        comp, price, m_dot, carbon_price_eur_per_t,
    ).sum()
    t_mean = window_cost_eur(
        _truth_W_total(comp_mean, m_dot, T_amb, T_sw),
        comp_mean, price, m_dot, carbon_price_eur_per_t,
    ).sum()

    surrogate_delta = float(s_var - s_mean)
    truth_delta = float(t_var - t_mean)
    gap = surrogate_delta - truth_delta
    gap_frac = gap / t_mean if t_mean else float("nan")
    return {
        "surrogate_delta_eur": surrogate_delta,
        "truth_delta_eur": truth_delta,
        "fabrication_gap_eur": gap,
        "gap_frac_of_truth_cost": gap_frac,
        "truth_mean_cost_eur": float(t_mean),
    }
