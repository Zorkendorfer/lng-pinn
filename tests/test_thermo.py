"""Tests for CoolProp thermo wrappers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.thermo import co2_per_kg_fuel, lower_heating_value, mixture_state

PURE_METHANE = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TYPICAL_LNG = (0.906, 0.063, 0.017, 0.004, 0.004, 0.006)  # US Gulf


def test_mixture_state_lng_density() -> None:
    # Compressed liquid LNG at ~111 K, 80 bar (post-pump): density 330–520 kg/m³
    state = mixture_state(TYPICAL_LNG, T=111.0, P=80e5)
    assert 330 < state.rho < 520, f"Unexpected density: {state.rho}"


def test_mixture_state_enthalpy_is_finite() -> None:
    state = mixture_state(TYPICAL_LNG, T=111.0, P=80e5)
    assert state.h == state.h  # not NaN
    assert abs(state.h) < 1e9


def test_lhv_methane() -> None:
    lhv = lower_heating_value(PURE_METHANE)
    # Pure methane LHV: ~802.3 kJ/mol
    assert 790_000 < lhv < 815_000, f"LHV out of range: {lhv}"


def test_lhv_mixture_greater_than_methane() -> None:
    lhv_mix = lower_heating_value(TYPICAL_LNG)
    lhv_ch4 = lower_heating_value(PURE_METHANE)
    # Heavier components have higher LHV, so mixture > pure methane
    assert lhv_mix > lhv_ch4


def test_composition_sums_to_one() -> None:
    total = sum(TYPICAL_LNG)
    assert abs(total - 1.0) < 1e-9


def test_co2_pure_methane() -> None:
    # CH4 + 2 O2 -> CO2 + 2 H2O. 1 mol CH4 (16.043 g) -> 1 mol CO2 (44.009 g).
    expected = 44.009 / 16.043  # ≈ 2.743
    got = co2_per_kg_fuel(PURE_METHANE)
    assert abs(got - expected) < 1e-3, f"CH4 CO2 factor {got} vs expected {expected}"


def test_co2_increases_with_heavies() -> None:
    """Heavier hydrocarbons have more carbon per kg → higher CO2 factor."""
    heavy = (0.82, 0.12, 0.035, 0.015, 0.010, 0.0)
    assert co2_per_kg_fuel(heavy) > co2_per_kg_fuel(PURE_METHANE)


def test_co2_in_natural_gas_envelope() -> None:
    """All compositions in the v1 operating envelope land in 2.5–3.0 kg CO2/kg."""
    for comp in (
        PURE_METHANE,
        TYPICAL_LNG,
        (0.82, 0.12, 0.035, 0.015, 0.010, 0.0),
        (0.96, 0.02, 0.005, 0.001, 0.001, 0.013),
    ):
        f = co2_per_kg_fuel(comp)
        assert 2.5 < f < 3.0, f"{comp}: CO2 factor {f} out of envelope"
