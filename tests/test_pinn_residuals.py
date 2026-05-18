"""Tests for PINN energy balance residuals on a trained checkpoint."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.pinn import RESULTS_DIR, Scaler, energy_balance_residual, load
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
def test_energy_residual_below_threshold() -> None:
    model, scaler = load(CHECKPOINT)
    model.eval()

    df = pd.read_parquet(TRAIN_SET).sample(n=100, random_state=0)
    if not {"h_in_per_kg", "h_out_per_kg"}.issubset(df.columns):
        pytest.skip("Training set must be rebuilt with v1.1 enthalpy columns")

    X_raw = torch.tensor(df[INPUT_COLS].values, dtype=torch.float32)
    X_col = scaler.scale_x(X_raw)
    h_in = torch.tensor(df["h_in_per_kg"].values, dtype=torch.float32)
    h_out = torch.tensor(df["h_out_per_kg"].values, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_col)
        residual = energy_balance_residual(X_col, y_pred, scaler, h_in, h_out)

    assert residual.item() < 0.01, (
        f"Energy balance residual {residual.item():.4f} exceeds 1% threshold"
    )


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="No trained checkpoint found")
def test_w_total_positive_on_random_inputs() -> None:
    model, scaler = load(CHECKPOINT)
    model.eval()

    torch.manual_seed(1)
    X = torch.randn(200, 9)
    with torch.no_grad():
        y_norm = model(X)
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
