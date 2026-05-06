"""CoolProp HEOS wrappers for LNG mixture thermodynamics.

Known limitations of HEOS backend:
- Reduced accuracy for heavy hydrocarbons (nC4, iC4) near critical point.
- Mixture interaction parameters from NIST; adequate for natural gas compositions.
- AbstractState objects are expensive to construct; cache by composition hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import CoolProp.CoolProp as CP

# Species in canonical order; mole fractions must sum to 1.
SPECIES = ("Methane", "Ethane", "Propane", "n-Butane", "IsoButane", "Nitrogen")
SPECIES_KEYS = ("CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2")


@dataclass(frozen=True)
class MixtureState:
    T: float  # K
    P: float  # Pa
    h: float  # J/mol  (molar enthalpy)
    s: float  # J/(mol·K)
    rho: float  # kg/m³


def _composition_key(x: tuple[float, ...]) -> str:
    return hashlib.md5(str(x).encode()).hexdigest()


@lru_cache(maxsize=512)
def _get_state(composition_key: str, x: tuple[float, ...]) -> Any:
    fluid_str = "&".join(SPECIES)
    state = CP.AbstractState("HEOS", fluid_str)
    state.set_mole_fractions(list(x))
    return state


def get_state(x: tuple[float, ...]) -> Any:
    """Return a cached AbstractState for the given mole-fraction tuple."""
    key = _composition_key(x)
    return _get_state(key, x)


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
