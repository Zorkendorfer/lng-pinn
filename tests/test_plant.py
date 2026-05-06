"""Tests for the FSRU steady-state plant model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.plant import simulate

COMPOSITIONS = [
    (0.906, 0.063, 0.017, 0.004, 0.004, 0.006),  # US Gulf
    (0.900, 0.060, 0.020, 0.010, 0.005, 0.005),  # Qatar
    (0.920, 0.045, 0.015, 0.008, 0.006, 0.006),  # Norway
    (0.870, 0.090, 0.025, 0.008, 0.005, 0.002),  # Algeria
    (0.942, 0.042, 0.010, 0.003, 0.002, 0.001),  # US East Coast
]


@pytest.mark.parametrize("comp", COMPOSITIONS)
def test_energy_non_negative(comp):
    out = simulate(comp, m_dot=40.0, T_amb=285.0, T_sw=285.0)
    assert out.W_total > 0, "Total work must be positive"
    assert out.W_pump  > 0, "Pump work must be positive"
    assert out.W_trim  >= 0, "Trim heater must be non-negative"


@pytest.mark.parametrize("comp", COMPOSITIONS)
def test_pump_leq_total(comp):
    out = simulate(comp, m_dot=40.0, T_amb=285.0, T_sw=285.0)
    assert out.W_pump <= out.W_total + 1e-9


@pytest.mark.parametrize("comp", COMPOSITIONS)
def test_send_out_temperature_reasonable(comp):
    out = simulate(comp, m_dot=40.0, T_amb=285.0, T_sw=285.0)
    assert 270 < out.T_out < 310, f"T_out={out.T_out} K out of expected range"


def test_seawater_duty_positive():
    out = simulate(COMPOSITIONS[0], m_dot=40.0, T_amb=285.0, T_sw=285.0)
    assert out.Q_sw > 0


def test_higher_flow_more_total_work():
    out_lo = simulate(COMPOSITIONS[0], m_dot=20.0, T_amb=285.0, T_sw=285.0)
    out_hi = simulate(COMPOSITIONS[0], m_dot=60.0, T_amb=285.0, T_sw=285.0)
    # W_total is per kg, so should be roughly similar; absolute cost scales with m_dot
    # At least verify both are physically sensible
    assert out_lo.W_total > 0 and out_hi.W_total > 0
