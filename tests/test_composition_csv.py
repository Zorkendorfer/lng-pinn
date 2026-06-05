"""Tests for the exogenous cargo-schedule loader (rework plan item 6)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.composition import COMP_COLS, build_composition_series_from_csv

US_GULF = (0.906, 0.063, 0.017, 0.004, 0.004, 0.006)
ALGERIA = (0.870, 0.090, 0.025, 0.008, 0.005, 0.002)


def _write_csv(tmp_path: Path) -> Path:
    rows = [
        {"arrival": "2024-01-01 00:00", **dict(zip(COMP_COLS, US_GULF))},
        {"arrival": "2024-01-13 00:00", **dict(zip(COMP_COLS, ALGERIA))},
    ]
    p = tmp_path / "cargoes.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_csv_series_on_simplex(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", "2024-01-31", freq="h", tz="UTC")
    df = build_composition_series_from_csv(idx, str(_write_csv(tmp_path)), blend_days=5)
    assert list(df.columns) == COMP_COLS
    assert (df.to_numpy() >= -1e-12).all()
    assert np.allclose(df.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_csv_blends_first_to_second(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", "2024-01-31", freq="h", tz="UTC")
    df = build_composition_series_from_csv(idx, str(_write_csv(tmp_path)), blend_days=5)
    # At the first arrival the tank holds the first cargo.
    assert abs(df["CH4"].iloc[0] - US_GULF[0]) < 1e-6
    # Five days after the second arrival the blend has fully settled to Algeria.
    settled = df.loc["2024-01-19 00:00":"2024-01-19 00:00", "CH4"].iloc[0]
    assert abs(settled - ALGERIA[0]) < 1e-6
    # Mid-transition (2 days into the 5-day blend) sits strictly between them.
    mid = df.loc["2024-01-15 00:00":"2024-01-15 00:00", "CH4"].iloc[0]
    assert ALGERIA[0] < mid < US_GULF[0]
