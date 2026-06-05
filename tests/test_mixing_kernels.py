"""Unit tests (rework plan item 1) for the tank-mixing kernels in
scripts/09_mixing_sensitivity.py.

Asserts that both blend kernels keep the composition on the mole-fraction
simplex (every fraction >= 0, every row sums to 1) at every hour, and that the
exponential kernel e-folds correctly. The script is loaded by path because its
module name starts with a digit.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "m09", ROOT / "scripts" / "09_mixing_sensitivity.py"
)
m09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m09)

from lng_pinn.composition import ARCHETYPES  # noqa: E402

KERNELS = ("linear", "exp", "step")
TAUS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)


def _arrivals() -> list:
    """A chained schedule: anchor + three cargo transitions across ~900 hours."""
    comps = [np.asarray(ARCHETYPES[k], dtype=float) for k in ARCHETYPES]
    return [
        (0, comps[0]),    # initial fill (anchor)
        (0, comps[1]),    # first transition at hour 0
        (288, comps[3]),  # one cargo cycle later
        (576, comps[2]),
    ]


def test_kernels_preserve_simplex() -> None:
    arrivals = _arrivals()
    n_hours = 900
    for kernel in KERNELS:
        for tau in TAUS:
            traj = m09.build_blended_trajectory(arrivals, n_hours, tau, kernel)
            assert traj.shape == (n_hours, 6)
            assert (traj >= -1e-12).all(), f"{kernel} tau={tau}: negative mole fraction"
            rowsum = traj.sum(axis=1)
            assert np.allclose(rowsum, 1.0, atol=1e-9), (
                f"{kernel} tau={tau}: max |rowsum-1| = {np.abs(rowsum - 1).max():.2e}"
            )


def test_blend_weight_in_unit_interval() -> None:
    dt = np.linspace(0.0, 30.0, 200)
    for kernel in KERNELS:
        w = m09.blend_weight(dt, 5.0, kernel)
        assert (w >= -1e-12).all() and (w <= 1.0 + 1e-12).all(), f"{kernel}: w out of [0,1]"


def test_exp_kernel_efolds() -> None:
    # At dt = tau the first-order kernel reaches 1 - 1/e of the transition.
    w = float(m09.blend_weight(np.array([5.0]), 5.0, "exp")[0])
    assert abs(w - (1.0 - np.exp(-1.0))) < 1e-12


def test_linear_kernel_endpoints() -> None:
    w = m09.blend_weight(np.array([0.0, 5.0, 10.0]), 5.0, "linear")
    assert w[0] == 0.0 and w[1] == 1.0 and w[2] == 1.0  # ramps over tau, then holds
