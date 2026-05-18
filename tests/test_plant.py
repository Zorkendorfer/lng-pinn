"""Tests for the FSRU steady-state plant model."""

import sys
from pathlib import Path
from typing import Any

import CoolProp.CoolProp as CP
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.plant import ETA_TRIM_HEATER, P_IN, P_OUT_DEFAULT, T_IN, T_SENDOUT, simulate
from lng_pinn.thermo import get_state

COMPOSITIONS = [
    (0.906, 0.063, 0.017, 0.004, 0.004, 0.006),  # US Gulf
    (0.900, 0.060, 0.020, 0.010, 0.005, 0.005),  # Qatar
    (0.920, 0.045, 0.015, 0.008, 0.006, 0.006),  # Norway
    (0.870, 0.090, 0.025, 0.008, 0.005, 0.002),  # Algeria
    (0.942, 0.042, 0.010, 0.003, 0.002, 0.001),  # US East Coast
]


@pytest.mark.parametrize("comp", COMPOSITIONS)
def test_energy_non_negative(comp: Any) -> None:
    out = simulate(comp, m_dot=40.0, T_amb=285.0, T_sw=285.0)
    assert out.W_total > 0, "Total work must be positive"
    assert out.W_pump > 0, "Pump work must be positive"
    assert out.W_trim >= 0, "Trim heater must be non-negative"


@pytest.mark.parametrize("comp", COMPOSITIONS)
def test_pump_leq_total(comp: Any) -> None:
    out = simulate(comp, m_dot=40.0, T_amb=285.0, T_sw=285.0)
    assert out.W_pump <= out.W_total + 1e-9


@pytest.mark.parametrize("comp", COMPOSITIONS)
def test_send_out_temperature_reasonable(comp: Any) -> None:
    out = simulate(comp, m_dot=40.0, T_amb=285.0, T_sw=285.0)
    assert 270 < out.T_out < 310, f"T_out={out.T_out} K out of expected range"


def test_seawater_duty_positive() -> None:
    out = simulate(COMPOSITIONS[0], m_dot=40.0, T_amb=285.0, T_sw=285.0)
    assert out.Q_sw > 0


def test_pump_eta_varies_with_flow() -> None:
    """Pump W_pump per kg is U-shaped: higher at low and high flows than at BEP (45 kg/s)."""
    comp = COMPOSITIONS[0]
    out_lo  = simulate(comp, m_dot=10.0, T_amb=285.0, T_sw=285.0)
    out_bep = simulate(comp, m_dot=45.0, T_amb=285.0, T_sw=285.0)
    out_hi  = simulate(comp, m_dot=80.0, T_amb=285.0, T_sw=285.0)
    # Efficiency is highest at BEP, so W_pump per kg is lowest there.
    assert out_lo.W_pump > out_bep.W_pump, "Low-flow W_pump/kg should exceed BEP"
    assert out_hi.W_pump > out_bep.W_pump, "High-flow W_pump/kg should exceed BEP"


def test_w_total_varies_with_flow() -> None:
    """W_total per kg must differ by > 1% between operating points - breaks bang-bang."""
    comp = COMPOSITIONS[0]
    out_lo  = simulate(comp, m_dot=10.0, T_amb=285.0, T_sw=285.0)
    out_bep = simulate(comp, m_dot=45.0, T_amb=285.0, T_sw=285.0)
    out_hi  = simulate(comp, m_dot=80.0, T_amb=285.0, T_sw=285.0)
    assert abs(out_lo.W_total - out_bep.W_total) / out_bep.W_total > 0.01
    assert abs(out_hi.W_total - out_bep.W_total) / out_bep.W_total > 0.01


def test_energy_balance_closes_within_half_percent() -> None:
    """Verify delta_h = W_pump + Q_sw + W_trim*eta to < 0.5% on a trim-active sample."""
    # Cold seawater forces the trim heater on (T_orv_out < T_SENDOUT).
    comp = COMPOSITIONS[0]
    T_sw = 275.0
    out  = simulate(comp, m_dot=40.0, T_amb=285.0, T_sw=T_sw)

    state = get_state(comp)
    state.update(CP.PT_INPUTS, P_IN, T_IN)
    h_in  = state.hmolar() / state.molar_mass()   # J/kg
    state.update(CP.PT_INPUTS, P_OUT_DEFAULT, T_SENDOUT)
    h_out = state.hmolar() / state.molar_mass()   # J/kg

    J_TO_KWH = 1.0 / 3_600_000.0
    delta_h  = (h_out - h_in) * J_TO_KWH         # kWh/kg

    rhs     = out.W_pump + out.Q_sw + out.W_trim * ETA_TRIM_HEATER
    rel_err = abs(delta_h - rhs) / abs(delta_h)

    assert rel_err < 0.005, f"Energy balance error {rel_err:.4%} exceeds 0.5%"
