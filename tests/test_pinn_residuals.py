"""Tests for PINN energy balance residuals on a trained checkpoint."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.pinn import RESULTS_DIR, Scaler, load
from lng_pinn.plant import ETA_TRIM_HEATER

CHECKPOINT = RESULTS_DIR / "pinn_v1.pt"
TRAIN_SET = Path("data/processed/train.parquet")
INPUT_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2", "m_dot", "T_amb", "T_sw"]


def test_scaler_to_moves_all_tensors() -> None:
    scaler = Scaler(*(torch.ones(2) for _ in range(4)))
    moved = scaler.to("cpu")

    assert all(tensor.device.type == "cpu" for tensor in moved)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="No trained checkpoint found")
@pytest.mark.skipif(not TRAIN_SET.exists(), reason="No training set found")
def test_v13_invariants() -> None:
    """v1.3 enforces two physical invariants exactly by construction:
      1. W_pump_pred == W_pump_expected (the analytical pump formula).
      2. W_total >= W_pump (trim heater can only add work, never remove it).
    """
    model, scaler = load(CHECKPOINT)
    model.eval()

    df = pd.read_parquet(TRAIN_SET).sample(n=200, random_state=0)
    if not {"h_in_per_kg", "h_out_per_kg", "W_pump_expected"}.issubset(df.columns):
        pytest.skip("Training set must be rebuilt with v1.1 enthalpy columns")

    X_raw = torch.tensor(df[INPUT_COLS].values, dtype=torch.float32)
    X_col = scaler.scale_x(X_raw)
    aux = torch.tensor(
        df[["h_in_per_kg", "h_out_per_kg", "W_pump_expected"]].values,
        dtype=torch.float32,
    )

    with torch.no_grad():
        y = scaler.unscale_y(model(X_col, aux, scaler=scaler))

    pump_err = (y[:, 0] - aux[:, 2]).abs().max().item()
    assert pump_err < 1e-5, f"W_pump deviates from analytical formula by {pump_err:.2e}"
    assert (y[:, 1] >= y[:, 0] - 1e-6).all(), "W_total < W_pump in some samples"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="No trained checkpoint found")
@pytest.mark.skipif(not TRAIN_SET.exists(), reason="No training set found")
def test_w_total_positive_on_random_inputs() -> None:
    """Sample real composition+flow rows from the training set rather than
    randn inputs (random gaussian compositions are non-physical and would
    fail the CoolProp aux lookup)."""
    model, scaler = load(CHECKPOINT)
    model.eval()

    df = pd.read_parquet(TRAIN_SET).sample(n=200, random_state=1)
    X_raw = torch.tensor(df[INPUT_COLS].values, dtype=torch.float32)
    aux = torch.tensor(
        df[["h_in_per_kg", "h_out_per_kg", "W_pump_expected"]].values,
        dtype=torch.float32,
    )

    with torch.no_grad():
        y_norm = model(scaler.scale_x(X_raw), aux, scaler=scaler)
        y = scaler.unscale_y(y_norm)

    W_total = y[:, 1]
    negative_frac = (W_total < 0).float().mean().item()
    assert negative_frac == 0.0, f"{negative_frac:.1%} of predictions have W_total < 0"


@pytest.mark.skipif(not TRAIN_SET.exists(), reason="No training set found")
def test_energy_balance_holds_on_training_data() -> None:
    df = pd.read_parquet(TRAIN_SET)
    if not {"h_in_per_kg", "h_out_per_kg"}.issubset(df.columns):
        pytest.skip("Training set must be rebuilt with v1.1 enthalpy columns")

    sample = df.sample(n=min(1000, len(df)), random_state=1)
    delta_h = sample["h_out_per_kg"] - sample["h_in_per_kg"]
    rhs = (sample["W_pump"] + sample["W_trim"] * ETA_TRIM_HEATER + sample["Q_sw"]) * 3.6e6
    rel_err = ((delta_h - rhs).abs() / delta_h.abs()).max()

    assert rel_err < 0.02, f"Max training-set energy balance error {rel_err:.4%} exceeds 2%"
