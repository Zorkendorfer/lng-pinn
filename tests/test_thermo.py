"""Tests for CoolProp thermo wrappers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.thermo import SPECIES, lower_heating_value, mixture_state

PURE_METHANE = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TYPICAL_LNG  = (0.906, 0.063, 0.017, 0.004, 0.004, 0.006)  # US Gulf


def test_mixture_state_methane_density():
    state = mixture_state(PURE_METHANE, T=111.0, P=1e5)
    # Liquid methane at ~111 K, 1 bar: density ~420–430 kg/m³
    assert 400 < state.rho < 460, f"Unexpected density: {state.rho}"


def test_mixture_state_enthalpy_is_finite():
    state = mixture_state(TYPICAL_LNG, T=111.0, P=80e5)
    assert state.h == state.h  # not NaN
    assert abs(state.h) < 1e9


def test_lhv_methane():
    lhv = lower_heating_value(PURE_METHANE)
    # Pure methane LHV: ~802.3 kJ/mol
    assert 790_000 < lhv < 815_000, f"LHV out of range: {lhv}"


def test_lhv_mixture_greater_than_methane():
    lhv_mix = lower_heating_value(TYPICAL_LNG)
    lhv_ch4 = lower_heating_value(PURE_METHANE)
    # Heavier components have higher LHV, so mixture > pure methane
    assert lhv_mix > lhv_ch4


def test_composition_sums_to_one():
    total = sum(TYPICAL_LNG)
    assert abs(total - 1.0) < 1e-9
