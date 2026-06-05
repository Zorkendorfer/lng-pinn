"""Train the v1.3 physics-constrained PINN surrogate.

The new architecture removes the physics-residual losses (energy balance,
pump work) — they are enforced exactly by construction. Training is now
pure supervised regression on (W_total, T_out, alpha, exergy), with
uncertainty-weighted multi-task loss handled inside ``pinn.train``.

The script supports ``--arch hard`` (default) and ``--arch soft``. For the hard
architecture, ``--lambda-e`` and ``--lambda-p`` remain compatibility flags; for
the soft architecture they are active physics-penalty weights.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.pinn import FINAL_PATH, SOFT_FINAL_PATH, Scaler, train

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
INPUT_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2", "m_dot", "T_amb", "T_sw"]
OUTPUT_COLS = ["W_pump", "W_total", "T_out", "exergy_destruction"]
AUX_COLS = ["h_in_per_kg", "h_out_per_kg", "W_pump_expected"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument(
        "--arch",
        choices=["hard", "soft"],
        default="hard",
        help="Surrogate architecture: hard physics-by-construction or soft penalty-loss baseline.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--lambda-c", type=float, default=1.0,
        help="Weight for the relative-cost loss term (A1).",
    )
    parser.add_argument("--patience", type=int, default=4_000, help="Early-stop patience in steps")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping (overrides --patience). Lets training run the full --steps.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing architecture-specific checkpoint and start from step 0.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Alias for --no-resume, kept for reproducibility command blocks.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Optional final checkpoint path. Training still uses the architecture "
            "default internally, then copies the final checkpoint here if different."
        ),
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=None,
        help="Save a full-state checkpoint every K steps (default: n_steps // 20).",
    )
    # Accepted-but-ignored v1.1 collocation args (kept so old commands don't break).
    parser.add_argument("--n-col", type=int, default=0)
    parser.add_argument("--lambda-e", type=float, default=0.0)
    parser.add_argument("--lambda-p", type=float, default=0.0)
    args = parser.parse_args()

    # `--no-early-stop` is sugar for an unreachable patience target.
    if args.no_early_stop:
        args.patience = args.steps + 1

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}  {args}")

    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    needed = INPUT_COLS + OUTPUT_COLS + AUX_COLS
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise SystemExit(
            f"Training set missing columns {missing}. "
            "Rebuild it with `uv run python scripts/02_build_dataset.py`."
        )

    # 80 / 10 / 10 split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train = int(0.8 * len(df))
    n_val = int(0.1 * len(df))
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train : n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val :]]
    print(f"Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    X_train_np = df_train[INPUT_COLS].values.astype(np.float32)
    y_train_np = df_train[OUTPUT_COLS].values.astype(np.float32)
    aux_train_np = df_train[AUX_COLS].values.astype(np.float32)

    x_mean = torch.tensor(X_train_np.mean(0), dtype=torch.float32)
    x_std = torch.tensor(X_train_np.std(0) + 1e-8, dtype=torch.float32)
    y_mean = torch.tensor(y_train_np.mean(0), dtype=torch.float32)
    y_std = torch.tensor(y_train_np.std(0) + 1e-8, dtype=torch.float32)
    scaler = Scaler(x_mean, x_std, y_mean, y_std)

    X_train = (torch.tensor(X_train_np) - x_mean) / x_std
    y_train = (torch.tensor(y_train_np) - y_mean) / y_std
    aux_train = torch.tensor(aux_train_np)

    X_val_np = df_val[INPUT_COLS].values.astype(np.float32)
    y_val_np = df_val[OUTPUT_COLS].values.astype(np.float32)
    aux_val_np = df_val[AUX_COLS].values.astype(np.float32)
    X_val = (torch.tensor(X_val_np) - x_mean) / x_std
    y_val = (torch.tensor(y_val_np) - y_mean) / y_std
    aux_val = torch.tensor(aux_val_np)

    model = train(
        X_train,
        y_train,
        aux_train,
        scaler,
        X_val=X_val,
        y_val=y_val,
        aux_val=aux_val,
        n_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        ema_decay=args.ema_decay,
        patience=args.patience,
        lambda_cost=args.lambda_c,
        resume=not (args.no_resume or args.fresh),
        ckpt_every=args.ckpt_every,
        lambda_energy=args.lambda_e,
        lambda_pump=args.lambda_p,
        arch=args.arch,
    )
    final_path = FINAL_PATH if args.arch == "hard" else SOFT_FINAL_PATH
    if args.out is not None:
        requested = Path(args.out)
        if requested != final_path:
            requested.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_path, requested)
            final_path = requested
    print(f"Checkpoint saved to {final_path}")

    # Evaluate on held-out test set
    X_test_np = df_test[INPUT_COLS].values.astype(np.float32)
    y_test_np = df_test[OUTPUT_COLS].values.astype(np.float32)
    aux_test_np = df_test[AUX_COLS].values.astype(np.float32)
    X_test = (torch.tensor(X_test_np) - x_mean) / x_std
    aux_test = torch.tensor(aux_test_np)

    model.eval()
    with torch.no_grad():
        y_pred_test = scaler.unscale_y(model(X_test, aux_test, scaler=scaler)).numpy()

    eval_records = []
    for i, col in enumerate(OUTPUT_COLS):
        y_true = y_test_np[:, i]
        y_pred = y_pred_test[:, i]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
        eval_records.append({"channel": col, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"  {col:24s}  MAE={mae:.5f}  RMSE={rmse:.5f}  R^2={r2:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = RESULTS_DIR / (
        "surrogate_eval.parquet" if args.arch == "hard" else "surrogate_eval_soft.parquet"
    )
    pd.DataFrame(eval_records).to_parquet(eval_path, index=False)
    print(f"Surrogate evaluation saved to {eval_path}")


if __name__ == "__main__":
    main()
