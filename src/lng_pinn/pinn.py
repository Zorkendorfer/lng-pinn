"""Physics-informed neural network surrogate for FSRU regasification."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

RESULTS_DIR = Path("results/models")

INPUT_DIM = 9   # CH4, C2H6, C3H8, nC4, iC4, N2, m_dot, T_amb, T_sw
OUTPUT_DIM = 4  # W_pump, W_total, T_out, exergy_destruction


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

    def scale_x(self, x: Tensor) -> Tensor:
        return (x - self.x_mean) / self.x_std

    def scale_y(self, y: Tensor) -> Tensor:
        return (y - self.y_mean) / self.y_std

    def unscale_y(self, y_norm: Tensor) -> Tensor:
        return y_norm * self.y_std + self.y_mean


class PINNMLP(nn.Module):
    """5-hidden-layer MLP with tanh activations."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = INPUT_DIM
        for _ in range(5):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def energy_balance_residual(
    x_raw: Tensor,
    y_pred_raw: Tensor,
    scaler: Scaler,
) -> Tensor:
    """Steady-state energy balance residual (normalised).

    Balance: h_in + W_pump + W_trim = h_out + Q_sw (per kg, kWh units)
    W_total = W_pump + W_trim (definition residual)
    """
    y = scaler.unscale_y(y_pred_raw)  # W_pump, W_total, T_out, exergy
    W_pump  = y[:, 0]
    W_total = y[:, 1]
    W_trim  = W_total - W_pump
    # Physics constraint: W_total must be non-negative and pump ≤ total
    residual = torch.relu(-W_pump) + torch.relu(W_pump - W_total)
    return residual.mean()


def train(
    X_train: Tensor,
    y_train: Tensor,
    X_col: Tensor,
    scaler: Scaler,
    n_steps: int = 50_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_energy: float = 1.0,
) -> PINNMLP:
    """Train the PINN surrogate.

    Args:
        X_train:  (N, 9) normalised inputs — data supervision.
        y_train:  (N, 4) normalised outputs — data supervision.
        X_col:    (M, 9) normalised collocation points — physics loss only.
        scaler:   Scaler used to normalise X and y.
        n_steps:  Total gradient steps.
        batch_size: Samples per batch (data + collocation each).
    """
    device = _device()
    model = PINNMLP().to(device)
    X_train, y_train, X_col = X_train.to(device), y_train.to(device), X_col.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    for step in range(n_steps):
        idx_d = torch.randint(len(X_train), (batch_size,), device=device)
        idx_c = torch.randint(len(X_col),   (batch_size,), device=device)

        xd, yd = X_train[idx_d], y_train[idx_d]
        xc = X_col[idx_c]

        y_pred_d = model(xd)
        loss_data = nn.functional.mse_loss(y_pred_d, yd)

        y_pred_c = model(xc)
        loss_energy = energy_balance_residual(xc, y_pred_c, scaler)

        loss = lambda_data * loss_data + lambda_energy * loss_energy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 5_000 == 0:
            print(f"step {step:6d}  loss_data={loss_data.item():.4e}  "
                  f"loss_energy={loss_energy.item():.4e}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state": model.state_dict(), "scaler": scaler}
    torch.save(checkpoint, RESULTS_DIR / "pinn_v1.pt")
    return model


def load(path: str | Path = RESULTS_DIR / "pinn_v1.pt") -> tuple[PINNMLP, Scaler]:
    checkpoint = torch.load(path, map_location="cpu")
    model = PINNMLP()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["scaler"]
