"""Regression test (rework plan item 5): the storage state must stay
liquid-phase across the whole operating composition envelope, especially the
high-N2 corner.

Background. ``composition_aux`` pins the storage state to the bubble point via
``PQ_INPUTS`` (Q=0). An earlier version used ``PT_INPUTS`` at a fixed
T_IN = 111 K, which returned a *vapour-phase* density for high-N2 mixtures whose
bubble point sits below 111 K, inflating pump work ~30x for roughly a quarter of
the training rows. This test fails if that regression returns: saturated-liquid
LNG is ~300-550 kg/m3, whereas vapour at 1 bar is only a few kg/m3.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.dataset import _sample_compositions
from lng_pinn.thermo import composition_aux

# Saturated-liquid LNG density band. Vapour at storage pressure is ~1-5 kg/m3,
# so any value inside this band is unambiguously liquid.
LIQUID_MIN = 300.0  # kg/m3
LIQUID_MAX = 550.0

# Highest reachable N2: all five hydrocarbons at their envelope minima, so the
# sum-to-one remainder (clipped to 0.02) is assigned to N2, then renormalised.
HIGH_N2 = tuple(float(v) for v in _sample_compositions(np.zeros((1, 5)))[0])


def _envelope_sample(n: int = 48, seed: int = 0) -> np.ndarray:
    """n compositions drawn uniformly over the LHS training envelope."""
    return _sample_compositions(LatinHypercube(d=5, seed=seed).random(n))


def test_high_n2_corner_is_liquid() -> None:
    assert HIGH_N2[5] > 0.02, f"expected the high-N2 corner, got N2={HIGH_N2[5]}"
    h_in, _h_out, rho_in = composition_aux(HIGH_N2)
    assert np.isfinite(h_in)
    assert LIQUID_MIN < rho_in < LIQUID_MAX, (
        f"high-N2 corner {tuple(round(v, 4) for v in HIGH_N2)}: storage density "
        f"{rho_in:.1f} kg/m3 is not liquid-phase (regression to PT_INPUTS at "
        f"fixed T_IN?)"
    )


def test_storage_density_liquid_across_envelope() -> None:
    bad = []
    for x in _envelope_sample():
        _, _, rho_in = composition_aux(tuple(float(v) for v in x))
        if not (LIQUID_MIN < rho_in < LIQUID_MAX):
            bad.append((tuple(round(v, 4) for v in x), round(rho_in, 1)))
    assert not bad, (
        f"{len(bad)} envelope points returned a non-liquid storage density "
        f"(vapour-phase regression): {bad[:5]}"
    )
