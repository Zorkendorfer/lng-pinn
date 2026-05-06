"""Train the PINN surrogate on the pre-built dataset."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.pinn import Scaler, train

PROCESSED_DIR = Path("data/processed")
INPUT_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2", "m_dot", "T_amb", "T_sw"]
OUTPUT_COLS = ["W_pump", "W_total", "T_out", "exergy_destruction"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--n-col", type=int, default=10_000, help="Number of physics collocation points"
    )
    parser.add_argument("--lambda-e", type=float, default=1.0)
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}  {args}")

    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    X_np = df[INPUT_COLS].values.astype(np.float32)
    y_np = df[OUTPUT_COLS].values.astype(np.float32)

    x_mean = torch.tensor(X_np.mean(0), dtype=torch.float32)
    x_std = torch.tensor(X_np.std(0) + 1e-8, dtype=torch.float32)
    y_mean = torch.tensor(y_np.mean(0), dtype=torch.float32)
    y_std = torch.tensor(y_np.std(0) + 1e-8, dtype=torch.float32)
    scaler = Scaler(x_mean, x_std, y_mean, y_std)

    X = (torch.tensor(X_np) - x_mean) / x_std
    y = (torch.tensor(y_np) - y_mean) / y_std

    # Collocation points: sample from marginal ranges of X (no CoolProp call)
    col_np = X_np[np.random.default_rng(99).choice(len(X_np), args.n_col, replace=True)]
    X_col = (torch.tensor(col_np) - x_mean) / x_std

    train(
        X,
        y,
        X_col,
        scaler,
        n_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        lambda_energy=args.lambda_e,
    )
    print("Checkpoint saved to results/models/pinn_v1.pt")


if __name__ == "__main__":
    main()
