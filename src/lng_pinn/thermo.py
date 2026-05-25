"""CoolProp HEOS wrappers for LNG mixture thermodynamics.

Known limitations of HEOS backend:
- Reduced accuracy for heavy hydrocarbons (nC4, iC4) near critical point.
- Mixture interaction parameters from NIST; adequate for natural gas compositions.
- AbstractState construction is expensive (~0.1s); use a singleton and call
  set_mole_fractions() before each update rather than constructing per composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import CoolProp.CoolProp as CP

# Species in canonical order; mole fractions must sum to 1.
SPECIES = ("Methane", "Ethane", "Propane", "n-Butane", "IsoButane", "Nitrogen")
SPECIES_KEYS = ("CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2")

# v1.3 B1: combustion stoichiometry for the carbon-cost dispatch term.
MW_CO2 = 44.009  # g/mol
MW_SPECIES = {  # g/mol — must align with SPECIES order below
    "Methane":   16.043,
    "Ethane":    30.070,
    "Propane":   44.097,
    "n-Butane":  58.123,
    "IsoButane": 58.123,
    "Nitrogen":  28.013,
}
_C_PER_MOL = (1, 2, 3, 4, 4, 0)  # carbons per molecule, aligned with SPECIES

_STATE: Any = None


def _fluid_str() -> str:
    return "&".join(SPECIES)


def get_state(x: tuple[float, ...]) -> Any:
    """Return the singleton AbstractState configured for composition x.

    The same C++ object is reused across calls; mole fractions are reset
    each time. Not thread-safe — adequate for single-threaded simulation.
    """
    global _STATE
    if _STATE is None:
        _STATE = CP.AbstractState("HEOS", _fluid_str())
    _STATE.set_mole_fractions(list(x))
    return _STATE


@dataclass(frozen=True)
class MixtureState:
    T: float  # K
    P: float  # Pa
    h: float  # J/mol  (molar enthalpy)
    s: float  # J/(mol·K)
    rho: float  # kg/m³


def mixture_state(x: tuple[float, ...], T: float, P: float) -> MixtureState:
    """Compute thermodynamic state for LNG mixture at given T, P."""
    state = get_state(x)
    state.update(CP.PT_INPUTS, P, T)
    return MixtureState(
        T=T,
        P=P,
        h=state.hmolar(),
        s=state.smolar(),
        rho=state.rhomass(),
    )


def composition_aux(composition: tuple[float, ...]) -> tuple[float, float, float]:
    """Return (h_in J/kg, h_out J/kg, rho_in kg/m^3) for one LNG composition.

    h_in is the saturated-liquid enthalpy at storage conditions (P_IN, T_IN);
    h_out is the gas enthalpy at send-out conditions (P_OUT_DEFAULT, T_SENDOUT);
    rho_in is the liquid density at storage conditions, needed for pump work.

    Cached per composition tuple (rounded to 6 decimals) so dispatch can
    request aux for many flow levels without re-querying CoolProp.
    """
    # Imported here to avoid a circular import with plant.py.
    from lng_pinn.plant import P_IN, P_OUT_DEFAULT, T_IN, T_SENDOUT

    key = tuple(round(v, 6) for v in composition)
    cached = _AUX_CACHE.get(key)
    if cached is not None:
        return cached
    state = get_state(composition)
    state.specify_phase(CP.iphase_liquid)
    state.update(CP.PT_INPUTS, P_IN, T_IN)
    h_in = state.hmolar() / state.molar_mass()
    rho_in = state.rhomass()
    state.unspecify_phase()
    state.update(CP.PT_INPUTS, P_OUT_DEFAULT, T_SENDOUT)
    h_out = state.hmolar() / state.molar_mass()
    result = (float(h_in), float(h_out), float(rho_in))
    _AUX_CACHE[key] = result
    return result


_AUX_CACHE: dict[tuple[float, ...], tuple[float, float, float]] = {}


def co2_per_kg_fuel(x: tuple[float, ...]) -> float:
    """kg of CO2 released per kg of fuel fully combusted (no slip, no flare).

    Used by the v1.3 B1 carbon-cost dispatch term. The mole-fraction vector
    ``x`` follows the canonical SPECIES order. Nitrogen contributes mass to
    the denominator but no carbon. For natural-gas compositions in the v1
    operating envelope this returns ~2.50–2.95 kg CO2/kg fuel — a ~15%
    spread that drives the composition signal at non-zero carbon prices.
    """
    mol_co2 = sum(xi * ci for xi, ci in zip(x, _C_PER_MOL))
    mw_fuel = sum(xi * MW_SPECIES[sp] for xi, sp in zip(x, SPECIES))
    return mol_co2 * MW_CO2 / mw_fuel


def lower_heating_value(x: tuple[float, ...]) -> float:
    """Return molar LHV (J/mol) of the mixture via component LHVs."""
    # LHV values (J/mol) from NIST/GPA at 25 °C, 1 atm
    lhv_components = {
        "Methane": 802_300.0,
        "Ethane": 1_427_800.0,
        "Propane": 2_043_100.0,
        "n-Butane": 2_657_400.0,
        "IsoButane": 2_651_400.0,
        "Nitrogen": 0.0,
    }
    return sum(xi * lhv_components[sp] for xi, sp in zip(x, SPECIES))
