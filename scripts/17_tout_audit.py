"""Audit the T_out output channel used in the surrogate fidelity table."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(PROCESSED_DIR / "train.parquet"))
    parser.add_argument("--eval", default=str(RESULTS_DIR / "surrogate_eval.csv"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "tout_audit.csv"))
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    eval_df = pd.read_csv(args.eval)
    if "T_out" not in train.columns:
        raise SystemExit(f"{args.train} has no T_out column")

    row = eval_df[eval_df["channel"].astype(str) == "T_out"]
    if row.empty:
        raise SystemExit(f"{args.eval} has no T_out row")

    t = train["T_out"].astype(float)
    mae = float(row.iloc[0]["MAE"])
    rmse = float(row.iloc[0]["RMSE"])
    r2 = float(row.iloc[0]["R2"])
    std = float(t.std(ddof=1))
    span = float(t.max() - t.min())
    degenerate = math.isclose(span, 0.0, abs_tol=1e-9) or std < 1e-9
    audit = pd.DataFrame(
        [
            {
                "channel": "T_out",
                "train_n": int(t.size),
                "train_mean_K": float(t.mean()),
                "train_std_K": std,
                "train_min_K": float(t.min()),
                "train_max_K": float(t.max()),
                "train_span_K": span,
                "eval_mae_K": mae,
                "eval_rmse_K": rmse,
                "eval_r2": r2,
                "mae_over_train_std": mae / std if std > 0 else float("inf"),
                "degenerate_setpoint_channel": bool(degenerate),
            }
        ]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    print(audit.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
