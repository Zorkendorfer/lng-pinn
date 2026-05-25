"""Physics-constrained neural surrogate for FSRU regasification.

Architecture rationale
----------------------
Three of the four target outputs (W_pump, W_trim, W_total, Q_sw, exergy)
are *redundant with closed-form thermodynamics* once the trim/seawater duty
split is known. The previous v1.1 design predicted all four with soft
penalty losses on energy balance and pump work, which left two pathologies:
  1. The network had to spend capacity learning a quantity (W_pump) that
     has an exact analytical form from composition and flow.
  2. Energy balance was only enforced softly, so W_total error was
     dominated by composition-uncorrelated noise — masking the composition
     signal the downstream dispatch optimisation is supposed to exploit.

The v1.3 architecture predicts only the genuinely-nonlinear quantities:
  - T_out: send-out temperature (free real-valued).
  - alpha in (0,1): fraction of vaporiser duty supplied by the electric
    trim heater (the remainder is "free" seawater heat).
  - exergy_destruction: diagnostic, kept as an output for the paper.

W_pump, W_trim, W_total, Q_sw are then derived in closed form using the
analytical pump-work formula and an exact enthalpy balance:

    delta_h_total = h_out(composition) - h_in(composition)     [J/kg]
    w_pump        = (P_out - P_in) / rho_in / eta_pump(m_dot)  [J/kg]
    Q_duty        = delta_h_total - w_pump                     [J/kg]
    Q_trim        = alpha * Q_duty
    Q_sw          = (1 - alpha) * Q_duty
    W_trim        = Q_trim / eta_trim_heater
    W_total       = w_pump + W_trim

Energy balance and pump-work residuals are therefore **zero by
construction**; the only learning signal is data fit on T_out, alpha
(implicit through W_total), and exergy. This gives the composition signal
a clean gradient path to the cost function used by dispatch.

Training improvements
---------------------
- SiLU activations + residual MLP blocks for smoother optimisation than
  the v1.1 plain tanh tower.
- Kendall multi-task uncertainty weights (learnable log-variances) for
  the three loss terms — replaces the unbalanced fixed lambdas.
- AdamW with cosine schedule + linear warm-up.
- Exponential moving average (EMA) of weights at decay 0.999; the EMA
  copy is what we save and evaluate.
- Wider hidden layer (256) — free on M-series MPS for this problem size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm

RESULTS_DIR = Path("results/models")
CKPT_PATH = RESULTS_DIR / "pinn_v1.ckpt"
FINAL_PATH = RESULTS_DIR / "pinn_v1.pt"

INPUT_DIM = 9   # CH4, C2H6, C3H8, nC4, iC4, N2, m_dot, T_amb, T_sw
OUTPUT_DIM = 4  # W_pump, W_total, T_out, exergy_destruction (publicly preserved)
AUX_DIM = 3     # h_in (J/kg), h_out (J/kg), W_pump_expected (kWh/kg)

ETA_TRIM_HEATER = 0.98
J_TO_KWH = 1.0 / 3_600_000.0


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


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        for layer in (self.lin1, self.lin2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.lin2(self.act(self.lin1(self.act(x))))


class PINNMLP(nn.Module):
    """Physics-constrained network: predicts (T_out, alpha, exergy);
    derives (W_pump, W_trim, W_total, Q_sw) from closed-form thermodynamics.

    Public ``forward`` returns the same 4-channel normalised output as v1.1
    (W_pump, W_total, T_out, exergy_destruction) so dispatch/baseline code
    keeps the same interface, but takes an additional ``aux`` tensor with
    composition-derived quantities (h_in, h_out, W_pump_expected) that
    the physics derivation needs.
    """

    def __init__(self, hidden: int = 256, n_blocks: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, hidden)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        self.blocks = nn.ModuleList(_ResidualBlock(hidden) for _ in range(n_blocks))
        # Head: 3 raw outputs (T_out_norm, alpha_logit, exergy_raw)
        self.head = nn.Linear(hidden, 3)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.act = nn.SiLU()
        self.softplus = nn.Softplus(beta=10.0)

        # Lower bound on exergy in normalised space (set via set_output_constraints).
        self.exergy_lower: Tensor
        self.register_buffer("exergy_lower", torch.tensor(-1.0e6))

    def set_output_constraints(self, scaler: Scaler) -> None:
        """Set normalised lower bound for exergy_destruction (channel 3 of y)."""
        # y_mean/y_std are length-4 vectors over (W_pump, W_total, T_out, exergy).
        lower = -scaler.y_mean[3] / scaler.y_std[3]
        self.exergy_lower.copy_(lower.to(self.exergy_lower.device))

    def _net_outputs(self, x_norm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the MLP. Returns (T_out_norm, alpha, exergy_norm)."""
        h = self.act(self.input_proj(x_norm))
        for block in self.blocks:
            h = block(h)
        raw = self.head(h)
        T_out_norm = raw[:, 0]
        alpha = torch.sigmoid(raw[:, 1])
        exergy_raw = raw[:, 2]
        exergy_norm = self.exergy_lower + self.softplus(exergy_raw - self.exergy_lower)
        return T_out_norm, alpha, exergy_norm

    def forward(
        self,
        x_norm: Tensor,
        aux: Tensor,
        scaler: Scaler | None = None,
    ) -> Tensor:
        """Return normalised (W_pump, W_total, T_out, exergy) of shape (B, 4).

        ``aux[:, 0]`` = h_in (J/kg), ``aux[:, 1]`` = h_out (J/kg),
        ``aux[:, 2]`` = W_pump_expected (kWh/kg).

        ``scaler`` is required so we can map raw physical W_pump/W_total to
        the normalised space the loss expects. We accept it as an argument
        rather than a buffer so train-time and inference-time scalers can
        be passed cleanly without mutating module state.
        """
        if scaler is None:
            scaler = self._cached_scaler  # type: ignore[attr-defined]

        T_out_norm, alpha, exergy_norm = self._net_outputs(x_norm)

        h_in = aux[:, 0]
        h_out = aux[:, 1]
        W_pump_kwh = aux[:, 2]
        # delta_h - w_pump = duty supplied by ORV + trim, in J/kg.
        delta_h = h_out - h_in
        W_pump_J = W_pump_kwh / J_TO_KWH  # kWh/kg -> J/kg
        duty_J = torch.clamp(delta_h - W_pump_J, min=0.0)
        W_trim_kwh = (alpha * duty_J / ETA_TRIM_HEATER) * J_TO_KWH
        W_total_kwh = W_pump_kwh + W_trim_kwh

        # Normalise the derived outputs back into the y-space the loss uses.
        W_pump_norm = (W_pump_kwh - scaler.y_mean[0]) / scaler.y_std[0]
        W_total_norm = (W_total_kwh - scaler.y_mean[1]) / scaler.y_std[1]

        return torch.stack([W_pump_norm, W_total_norm, T_out_norm, exergy_norm], dim=1)

    def attach_scaler(self, scaler: Scaler) -> None:
        """Cache a scaler so external callers can use forward(x, aux) without passing it."""
        self._cached_scaler = scaler  # type: ignore[attr-defined]


def energy_balance_residual(
    x_raw: Tensor,
    y_pred_raw: Tensor,
    scaler: Scaler,
    h_in: Tensor,
    h_out: Tensor,
) -> Tensor:
    """Steady-state energy-balance residual.

    Retained for backward compatibility with v1.1 tests and external code.
    The v1.3 architecture enforces this exactly by construction, so this
    function should always return ~0 on a trained model — it is now a
    sanity check rather than a training loss.
    """
    y = scaler.unscale_y(y_pred_raw)
    W_pump = y[:, 0] * 3.6e6
    W_total = y[:, 1] * 3.6e6
    W_trim = W_total - W_pump
    W_trim_heat = W_trim * ETA_TRIM_HEATER
    delta_h = h_out - h_in
    Q_sw_implied = delta_h - W_pump - W_trim_heat
    return torch.relu(-Q_sw_implied / 5e5).pow(2).mean()


def relative_cost_loss(
    y_pred_raw: Tensor,
    y_true_raw: Tensor,
    scaler: Scaler,
    eps: float = 1e-6,
) -> Tensor:
    """Relative-error MSE on W_pump and W_total (channels 0, 1).

    Aligns the training objective with the cost residual the dispatch uses:
        (cost_pred - cost_true) / cost_true = (W_pred - W_true) / W_true
    Price and m_dot cancel, so a relative-error MSE on W is exactly the
    relative-error MSE on dispatch cost.
    """
    y_pred = scaler.unscale_y(y_pred_raw)
    y_true = scaler.unscale_y(y_true_raw)
    w_pred = y_pred[:, [0, 1]]
    w_true = y_true[:, [0, 1]]
    return ((w_pred - w_true) / (w_true.abs() + eps)).pow(2).mean()


def _alpha_target(
    h_in: Tensor,
    h_out: Tensor,
    W_pump_kwh: Tensor,
    W_trim_kwh: Tensor,
) -> Tensor:
    """Recover the alpha = Q_trim / (delta_h - W_pump) target from training labels.

    Clamped to [0, 1]; samples where (delta_h - W_pump) is near zero get
    alpha = 0 (negligible duty case).
    """
    delta_h = h_out - h_in
    duty_J = delta_h - W_pump_kwh / J_TO_KWH
    Q_trim_J = W_trim_kwh / J_TO_KWH * ETA_TRIM_HEATER
    alpha = torch.where(duty_J > 1e3, Q_trim_J / duty_J, torch.zeros_like(duty_J))
    return torch.clamp(alpha, 0.0, 1.0)


class _EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module) -> dict[str, Tensor]:
        """Swap model state with EMA shadow; return the original state for restore."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return backup


def _save_ckpt(
    path: Path,
    *,
    model: PINNMLP,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: Scaler,
    step: int,
    n_steps: int,
    best_val_loss: float,
    best_state: dict[str, Tensor] | None,
    steps_since_improvement: int,
    done: bool,
    device: torch.device,
    log_var: Tensor | None = None,
    ema_shadow: dict[str, Tensor] | None = None,
) -> None:
    """Atomic write of the full training-state checkpoint.

    ``log_var`` and ``ema_shadow`` are v1.3 additions: the Kendall multi-task
    weights and the EMA model copy. Saving them makes resume bit-exact rather
    than just step-exact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {
        "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler": scaler.to("cpu"),
        "step": step,
        "n_steps": n_steps,
        "best_val_loss": best_val_loss,
        "best_state": best_state,
        "steps_since_improvement": steps_since_improvement,
        "done": done,
        "torch_rng_state": torch.get_rng_state(),
    }
    if log_var is not None:
        ckpt["log_var"] = log_var.detach().cpu().clone()
    if ema_shadow is not None:
        ckpt["ema_shadow"] = {k: v.detach().cpu().clone() for k, v in ema_shadow.items()}
    if device.type == "cuda":
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
    tmp = path.with_suffix(".ckpt.tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def _load_ckpt(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a training-state checkpoint. weights_only=False because we trust our own file."""
    with torch.serialization.safe_globals([Scaler]):
        return torch.load(path, map_location=device, weights_only=False)


def train(
    X_train: Tensor,
    y_train: Tensor,
    aux_train: Tensor,
    scaler: Scaler,
    X_val: Tensor | None = None,
    y_val: Tensor | None = None,
    aux_val: Tensor | None = None,
    # Legacy args (collocation + physics weights) accepted for backward compat
    # with scripts/03_train_pinn.py; they are unused by the physics-constrained
    # architecture but kept so old callers don't break.
    X_col: Tensor | None = None,
    h_in_col: Tensor | None = None,
    h_out_col: Tensor | None = None,
    W_pump_expected: Tensor | None = None,
    lambda_data: float = 1.0,
    lambda_energy: float = 0.0,
    lambda_pump: float = 0.0,
    n_steps: int = 50_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    warmup_steps: int = 1_000,
    val_every: int = 500,
    patience: int = 4_000,
    ema_decay: float = 0.999,
    lambda_cost: float = 1.0,
    resume: bool = True,
    ckpt_every: int | None = None,
) -> PINNMLP:
    """Train the v1.3 physics-constrained PINN.

    Inputs / labels are normalised (length-9 / length-4 respectively).
    ``aux_train`` has shape (N, 3): (h_in J/kg, h_out J/kg, W_pump_expected kWh/kg).

    ``lambda_cost`` controls the v1.3 A1 relative-cost loss, applied on top of
    the Kendall-weighted multi-task loss. The cost loss penalises relative
    error on W_pump and W_total — directly aligned with the dispatch objective.

    ``resume`` / ``ckpt_every`` enable two-level training: a long run can be
    resumed from the last full-state checkpoint (model + optimiser +
    scheduler + EMA + log_var + RNG) without losing progress.
    """
    del X_col, h_in_col, h_out_col, W_pump_expected, lambda_energy, lambda_pump
    del lambda_data  # all losses are now uncertainty-weighted

    device = _device()
    model = PINNMLP().to(device)
    model.set_output_constraints(scaler.to(device))
    scaler = scaler.to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    aux_train = aux_train.to(device)
    if X_val is not None:
        X_val = X_val.to(device)
        y_val = y_val.to(device)  # type: ignore[union-attr]
        aux_val = aux_val.to(device)  # type: ignore[union-attr]

    # Pre-compute the alpha supervision target from training labels.
    y_train_raw = scaler.unscale_y(y_train)
    W_pump_raw = y_train_raw[:, 0]
    W_total_raw = y_train_raw[:, 1]
    W_trim_raw = W_total_raw - W_pump_raw
    alpha_train = _alpha_target(aux_train[:, 0], aux_train[:, 1], W_pump_raw, W_trim_raw)

    # ---- Optimiser, scheduler, multi-task weights, EMA ----------------------------
    # Kendall multi-task uncertainty weights (one per task: alpha, W_total, T_out, exergy).
    # L = sum_k 0.5 * exp(-s_k) * loss_k + 0.5 * s_k.
    log_var = nn.Parameter(torch.zeros(4, device=device))

    optimizer = optim.AdamW(
        [*model.parameters(), log_var],
        lr=lr,
        weight_decay=weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.02 + 0.98 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = _EMA(model, decay=ema_decay)

    if ckpt_every is None:
        ckpt_every = max(1, n_steps // 20)

    start_step = 0
    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    steps_since_improve = 0

    # ---- Resume from checkpoint if present ----------------------------------------
    if resume and CKPT_PATH.exists():
        try:
            ckpt = _load_ckpt(CKPT_PATH, device)
            stored_n_steps = ckpt.get("n_steps", n_steps)
            if stored_n_steps != n_steps:
                print(
                    f"  WARN: checkpoint was set up for {stored_n_steps} total steps, "
                    f"now requested {n_steps}. Scheduler curve will not match a "
                    "from-scratch run exactly."
                )
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            if "log_var" in ckpt:
                log_var.data.copy_(ckpt["log_var"].to(device))
            if "ema_shadow" in ckpt:
                ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
            start_step = int(ckpt["step"])
            best_val = float(ckpt["best_val_loss"])
            best_state = ckpt.get("best_state")
            steps_since_improve = int(ckpt.get("steps_since_improvement", 0))
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
            if device.type == "cuda" and "cuda_rng_state" in ckpt:
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
            print(
                f"  Resumed from {CKPT_PATH.name}: step {start_step}/{n_steps}, "
                f"best val_loss={best_val:.4e}, since_improvement={steps_since_improve}"
            )
            if ckpt.get("done") or start_step >= n_steps:
                print("  Checkpoint marks training complete — restoring best weights.")
                if best_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                else:
                    ema.apply_to(model)
                model = model.cpu()
                scaler_cpu = scaler.to("cpu")
                model.set_output_constraints(scaler_cpu)
                model.attach_scaler(scaler_cpu)
                FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model_state": model.state_dict(), "scaler": scaler_cpu, "version": "v1.3"},
                    FINAL_PATH,
                )
                return model
        except Exception as exc:
            print(f"  Could not resume from {CKPT_PATH.name}: {exc}; starting from step 0")
            start_step = 0
            best_val = float("inf")
            best_state = None
            steps_since_improve = 0

    # ---- Training loop ------------------------------------------------------------
    early_stopped = False
    pbar = tqdm(
        range(start_step, n_steps),
        desc="Training PINN v1.3",
        unit="step",
        initial=start_step,
        total=n_steps,
    )
    last_step = start_step
    for step in pbar:
        last_step = step
        idx = torch.randint(len(X_train), (batch_size,), device=device)
        xb = X_train[idx]
        yb = y_train[idx]
        ab = aux_train[idx]
        alpha_b = alpha_train[idx]

        T_out_norm, alpha_pred, exergy_norm = model._net_outputs(xb)  # noqa: SLF001
        y_pred = model(xb, ab, scaler=scaler)

        loss_alpha = (alpha_pred - alpha_b).pow(2).mean()
        loss_W = (y_pred[:, 1] - yb[:, 1]).pow(2).mean()
        loss_T = (T_out_norm - yb[:, 2]).pow(2).mean()
        loss_E = (exergy_norm - yb[:, 3]).pow(2).mean()
        losses = torch.stack([loss_alpha, loss_W, loss_T, loss_E])

        # Kendall multi-task uncertainty weighting.
        loss_mt = (0.5 * torch.exp(-log_var) * losses + 0.5 * log_var).sum()

        # A1: relative-cost loss, applied directly (not uncertainty-weighted) so
        # it acts as a calibration term independent of the Kendall weights drifting.
        if lambda_cost > 0:
            loss_cost = relative_cost_loss(y_pred, yb, scaler)
        else:
            loss_cost = torch.zeros((), device=device)
        loss = loss_mt + lambda_cost * loss_cost

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if step % 500 == 0:
            with torch.no_grad():
                w = torch.exp(-log_var).tolist()
            pbar.set_postfix(
                a=f"{loss_alpha.item():.2e}",
                W=f"{loss_W.item():.2e}",
                T=f"{loss_T.item():.2e}",
                E=f"{loss_E.item():.2e}",
                c=f"{loss_cost.item():.2e}",
                wα=f"{w[0]:.2f}",
                wW=f"{w[1]:.2f}",
            )

        if X_val is not None and step > 0 and step % val_every == 0:
            backup = ema.apply_to(model)
            with torch.no_grad():
                y_pred_v = model(X_val, aux_val, scaler=scaler)  # type: ignore[arg-type]
                val_loss = (y_pred_v[:, 1] - y_val[:, 1]).pow(2).mean().item()  # type: ignore[index]
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                steps_since_improve = 0
            else:
                steps_since_improve += val_every
            model.load_state_dict(backup)
            if steps_since_improve >= patience:
                pbar.set_description(f"Early stop at step {step} (val={best_val:.3e})")
                early_stopped = True
                break

        # Periodic full-state checkpoint for resume.
        if (step + 1) % ckpt_every == 0:
            _save_ckpt(
                CKPT_PATH,
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                step=step + 1, n_steps=n_steps,
                best_val_loss=best_val, best_state=best_state,
                steps_since_improvement=steps_since_improve,
                done=False, device=device,
                log_var=log_var, ema_shadow=ema.shadow,
            )

    # ---- Final weights: EMA vs best-by-val, pick the lower ------------------------
    ema.apply_to(model)
    if best_state is not None and X_val is not None:
        with torch.no_grad():
            y_pred_v = model(X_val, aux_val, scaler=scaler)  # type: ignore[arg-type]
            ema_val = (y_pred_v[:, 1] - y_val[:, 1]).pow(2).mean().item()  # type: ignore[index]
        if best_val < ema_val:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final-state checkpoint marks completion.
    _save_ckpt(
        CKPT_PATH,
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        step=last_step + 1, n_steps=n_steps,
        best_val_loss=best_val, best_state=best_state,
        steps_since_improvement=steps_since_improve,
        done=True, device=device,
        log_var=log_var, ema_shadow=ema.shadow,
    )

    # Inference-only checkpoint for downstream scripts.
    model = model.cpu()
    scaler_cpu = scaler.to("cpu")
    model.set_output_constraints(scaler_cpu)
    model.attach_scaler(scaler_cpu)
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "scaler": scaler_cpu, "version": "v1.3"},
        FINAL_PATH,
    )
    if early_stopped:
        print(f"  Early stop fired at step {last_step}; checkpoint marked done.")
    return model


def build_aux(
    comp: np.ndarray | Tensor,
    m_dot: np.ndarray | Tensor,
) -> Tensor:
    """Build the (B, 3) aux tensor required by ``PINNMLP.forward``.

    ``comp`` has shape (B, 6) — mole fractions in canonical species order.
    ``m_dot`` has shape (B,) — kg/s.

    Returns a float32 CPU tensor: columns (h_in J/kg, h_out J/kg,
    W_pump_expected kWh/kg). Uses the cached CoolProp composition lookup
    so calling this with many flow levels at the same composition is
    cheap (one HEOS state init per unique composition).
    """
    from lng_pinn.plant import P_IN, P_OUT_DEFAULT, pump_efficiency
    from lng_pinn.thermo import composition_aux

    comp_arr = comp.detach().cpu().numpy() if isinstance(comp, Tensor) else np.asarray(comp)
    m_arr = m_dot.detach().cpu().numpy() if isinstance(m_dot, Tensor) else np.asarray(m_dot)
    n = comp_arr.shape[0]
    aux = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        h_in, h_out, rho_in = composition_aux(tuple(float(v) for v in comp_arr[i]))
        eta = pump_efficiency(float(m_arr[i]))
        w_pump_kwh = (P_OUT_DEFAULT - P_IN) / rho_in / eta * J_TO_KWH
        aux[i, 0] = h_in
        aux[i, 1] = h_out
        aux[i, 2] = w_pump_kwh
    return torch.from_numpy(aux)


def load(path: str | Path = FINAL_PATH) -> tuple[PINNMLP, Scaler]:
    with torch.serialization.safe_globals([Scaler]):
        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    scaler: Scaler = checkpoint["scaler"]
    model = PINNMLP()
    model.set_output_constraints(scaler)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.attach_scaler(scaler)
    model.eval()
    return model, scaler
