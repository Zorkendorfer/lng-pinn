"""Physics-informed neural network surrogate for FSRU regasification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm

RESULTS_DIR = Path("results/models")

INPUT_DIM = 9  # CH4, C2H6, C3H8, nC4, iC4, N2, m_dot, T_amb, T_sw
OUTPUT_DIM = 4  # W_pump, W_total, T_out, exergy_destruction

# Must match plant.py
ETA_TRIM_HEATER = 0.98


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Scaler(NamedTuple):
    x_mean: Tensor
    x_std: Tensor
    y_mean: Tensor
    y_std: Tensor

    def to(self, device: torch.device | str) -> "Scaler":
        return Scaler(
            self.x_mean.to(device),
            self.x_std.to(device),
            self.y_mean.to(device),
            self.y_std.to(device),
        )

    def scale_x(self, x: Tensor) -> Tensor:
        return (x - self.x_mean) / self.x_std

    def scale_y(self, y: Tensor) -> Tensor:
        return (y - self.y_mean) / self.y_std

    def unscale_y(self, y_norm: Tensor) -> Tensor:
        return y_norm * self.y_std + self.y_mean


class PINNMLP(nn.Module):
    """5-hidden-layer MLP with tanh activations.

    Channels 0 (W_pump), 1 (W_total), and 3 (exergy_destruction) are constrained
    to be non-negative in physical units by enforcing their scaler-implied lower
    bounds in normalised output space. T_out (channel 2) is left unconstrained.
    """

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = INPUT_DIM
        for _ in range(5):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus(beta=10.0)
        self.positive_output_lower: Tensor
        self.register_buffer("positive_output_lower", torch.full((3,), -1.0e6))

    def set_output_constraints(self, scaler: Scaler) -> None:
        """Set normalised lower bounds that correspond to zero physical output."""
        positive_channels = torch.tensor([0, 1, 3])
        lower = -scaler.y_mean[positive_channels] / scaler.y_std[positive_channels]
        self.positive_output_lower.copy_(lower.to(self.positive_output_lower.device))

    def forward(self, x: Tensor) -> Tensor:
        raw = self.net(x)
        positive_raw = raw[:, [0, 1, 3]]
        positive = self.positive_output_lower + self.softplus(
            positive_raw - self.positive_output_lower
        )
        return torch.stack(
            [
                positive[:, 0],  # W_pump >= 0 after unscaling
                positive[:, 1],  # W_total >= 0 after unscaling
                raw[:, 2],       # T_out free
                positive[:, 2],  # exergy >= 0 after unscaling
            ],
            dim=1,
        )


def energy_balance_residual(
    x_raw: Tensor,
    y_pred_raw: Tensor,
    scaler: Scaler,
    h_in: Tensor,   # (B,) J/kg - storage enthalpy per kg
    h_out: Tensor,  # (B,) J/kg - send-out enthalpy per kg
) -> Tensor:
    """Steady-state energy balance: delta_h = W_pump + W_trim*eta + Q_sw.

    Q_sw_implied must be non-negative (seawater is a heat source, not a sink).
    Normalised by typical |delta_h| ~ 5e5 J/kg for scale-invariance.
    """
    y = scaler.unscale_y(y_pred_raw)
    W_pump  = y[:, 0] * 3.6e6   # kWh/kg -> J/kg
    W_total = y[:, 1] * 3.6e6
    W_trim  = W_total - W_pump
    W_trim_heat = W_trim * ETA_TRIM_HEATER
    delta_h = h_out - h_in
    Q_sw_implied = delta_h - W_pump - W_trim_heat
    return torch.relu(-Q_sw_implied / 5e5).pow(2).mean()


def pump_work_residual(
    y_pred_raw: Tensor,
    scaler: Scaler,
    W_pump_expected: Tensor,  # (B,) kWh/kg - analytical pump work
) -> Tensor:
    """Penalise deviation from the analytical incompressible-pump work formula."""
    y = scaler.unscale_y(y_pred_raw)
    return ((y[:, 0] - W_pump_expected) / (W_pump_expected + 1e-8)).pow(2).mean()


def train(
    X_train: Tensor,
    y_train: Tensor,
    X_col: Tensor,
    h_in_col: Tensor,
    h_out_col: Tensor,
    W_pump_expected: Tensor,
    scaler: Scaler,
    X_val: Tensor | None = None,
    y_val: Tensor | None = None,
    n_steps: int = 50_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_energy: float = 1.0,
    lambda_pump: float = 1.0,
    val_every: int = 500,
    patience: int = 2000,
) -> PINNMLP:
    """Train the PINN surrogate.

    Args:
        X_train:          (N, 9) normalised inputs - data supervision.
        y_train:          (N, 4) normalised outputs - data supervision.
        X_col:            (M, 9) normalised collocation points - physics loss only.
        h_in_col:         (M,) J/kg - storage enthalpy for each collocation point.
        h_out_col:        (M,) J/kg - send-out enthalpy for each collocation point.
        W_pump_expected:  (N,) kWh/kg - analytical pump work for training points.
        scaler:           Scaler used to normalise X and y.
        X_val, y_val:     Optional validation set for early stopping.
        patience:         Steps without val improvement before stopping.
    """
    device = _device()
    model = PINNMLP()
    model.set_output_constraints(scaler)
    model = model.to(device)
    X_train     = X_train.to(device)
    y_train     = y_train.to(device)
    X_col       = X_col.to(device)
    h_in_col    = h_in_col.to(device)
    h_out_col   = h_out_col.to(device)
    W_pump_expected = W_pump_expected.to(device)
    scaler      = scaler.to(device)
    if X_val is not None:
        X_val = X_val.to(device)
        y_val = y_val.to(device)  # type: ignore[union-attr]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    steps_since_improvement = 0

    pbar = tqdm(range(n_steps), desc="Training PINN", unit="step")
    for step in pbar:
        idx_d = torch.randint(len(X_train), (batch_size,), device=device)
        idx_c = torch.randint(len(X_col),   (batch_size,), device=device)

        xd, yd = X_train[idx_d], y_train[idx_d]
        xc     = X_col[idx_c]
        h_in_b  = h_in_col[idx_c]
        h_out_b = h_out_col[idx_c]
        W_pump_b = W_pump_expected[idx_d]

        y_pred_d = model(xd)
        loss_data = nn.functional.mse_loss(y_pred_d, yd)

        y_pred_c = model(xc)
        loss_energy = energy_balance_residual(xc, y_pred_c, scaler, h_in_b, h_out_b)
        loss_pump   = pump_work_residual(y_pred_d, scaler, W_pump_b)

        loss = lambda_data * loss_data + lambda_energy * loss_energy + lambda_pump * loss_pump
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            pbar.set_postfix(
                data=f"{loss_data.item():.3e}",
                phys=f"{loss_energy.item():.3e}",
                pump=f"{loss_pump.item():.3e}",
            )

        # Early stopping on validation set
        if X_val is not None and step % val_every == 0:
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(X_val), y_val).item()  # type: ignore[arg-type]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                steps_since_improvement = 0
            else:
                steps_since_improvement += val_every
            if steps_since_improvement >= patience:
                pbar.set_description(f"Early stop at step {step}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Move everything to CPU before saving so the checkpoint is device-agnostic.
    model = model.cpu()
    scaler = scaler.to("cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state": model.state_dict(), "scaler": scaler}
    torch.save(checkpoint, RESULTS_DIR / "pinn_v1.pt")
    return model


def load(path: str | Path = RESULTS_DIR / "pinn_v1.pt") -> tuple[PINNMLP, Scaler]:
    with torch.serialization.safe_globals([Scaler]):
        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu")
    scaler: Scaler = checkpoint["scaler"]
    model = PINNMLP()
    model.set_output_constraints(scaler)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, scaler
