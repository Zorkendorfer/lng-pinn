"""Tests for PINN energy balance residuals on a trained checkpoint."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.pinn import RESULTS_DIR, energy_balance_residual, load

CHECKPOINT = RESULTS_DIR / "pinn_v1.pt"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="No trained checkpoint found")
def test_energy_residual_below_threshold() -> None:
    model, scaler = load(CHECKPOINT)
    model.eval()

    # 100 random inputs drawn from unit-normal (scaler space)
    torch.manual_seed(0)
    X_col = torch.randn(100, 9)

    with torch.no_grad():
        y_pred = model(X_col)
        residual = energy_balance_residual(X_col, y_pred, scaler)

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
    assert negative_frac < 0.05, f"{negative_frac:.1%} of predictions have W_total < 0"
