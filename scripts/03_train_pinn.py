"""Train the PINN surrogate on the pre-built dataset."""

import argparse
import subprocess
import sys
from pathlib import Path

import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.dataset import BOUNDS, _sample_compositions
from lng_pinn.pinn import Scaler, train
from lng_pinn.plant import P_IN, P_OUT_DEFAULT, T_IN, T_SENDOUT
from lng_pinn.thermo import get_state

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
INPUT_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2", "m_dot", "T_amb", "T_sw"]
OUTPUT_COLS = ["W_pump", "W_total", "T_out", "exergy_destruction"]
PUMP_COL = "W_pump_expected"
PHYSICS_COLS = ["h_in_per_kg", "h_out_per_kg", PUMP_COL]


def _compute_h_in_out_col(
    comp_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute h_in, h_out (J/kg) and a valid boolean mask for each collocation composition.

    Compositions that are not liquid at storage conditions (P_IN, T_IN) are marked invalid
    and get placeholder zeros; the caller should filter them out of X_col.
    """
    h_in_list, h_out_list, valid_list = [], [], []
    for comp in tqdm(comp_np, desc="h_in/h_out for collocation", unit="pts"):
        try:
            state = get_state(tuple(float(v) for v in comp))
            state.specify_phase(CP.iphase_liquid)
            state.update(CP.PT_INPUTS, P_IN, T_IN)
            h_in = state.hmolar() / state.molar_mass()
            state.unspecify_phase()
            state.update(CP.PT_INPUTS, P_OUT_DEFAULT, T_SENDOUT)
            h_out = state.hmolar() / state.molar_mass()
            h_in_list.append(h_in)
            h_out_list.append(h_out)
            valid_list.append(True)
        except Exception:
            h_in_list.append(0.0)
            h_out_list.append(0.0)
            valid_list.append(False)
    return (
        np.array(h_in_list, dtype=np.float32),
        np.array(h_out_list, dtype=np.float32),
        np.array(valid_list, dtype=bool),
    )


COLL_LHS_SEED = 99  # collocation-point LHS seed; cache is keyed by (n_col, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--n-col",
        type=int,
        default=10_000,
        help="Number of physics collocation points (LHS, no labels)",
    )
    parser.add_argument(
        "--lambda-e", type=float, default=1.0, help="Weight for energy-balance physics loss"
    )
    parser.add_argument(
        "--lambda-p", type=float, default=1.0, help="Weight for pump-work physics loss"
    )
    parser.add_argument("--patience", type=int, default=2000, help="Early-stop patience in steps")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping (overrides --patience). Lets training run the full --steps.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing pinn_v1.ckpt and start from step 0.",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=None,
        help="Save a full-state checkpoint every K steps (default: n_steps // 20).",
    )
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
    missing_cols = sorted(set(INPUT_COLS + OUTPUT_COLS + PHYSICS_COLS) - set(df.columns))
    if missing_cols:
        raise SystemExit(
            "Training set is missing v1.1 columns "
            f"{missing_cols}. Rebuild it with `uv run python scripts/02_build_dataset.py`."
        )

    # -- 80 / 10 / 10 train / val / test split ----------------------------------
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train = int(0.8 * len(df))
    n_val = int(0.1 * len(df))
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train : n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val :]]
    print(f"Split: {len(df_train)} train / {len(df_val)} val / {len(df_test)} test")

    # -- Scaler from training set only ------------------------------------------
    X_train_np = df_train[INPUT_COLS].values.astype(np.float32)
    y_train_np = df_train[OUTPUT_COLS].values.astype(np.float32)

    x_mean = torch.tensor(X_train_np.mean(0), dtype=torch.float32)
    x_std = torch.tensor(X_train_np.std(0) + 1e-8, dtype=torch.float32)
    y_mean = torch.tensor(y_train_np.mean(0), dtype=torch.float32)
    y_std = torch.tensor(y_train_np.std(0) + 1e-8, dtype=torch.float32)
    scaler = Scaler(x_mean, x_std, y_mean, y_std)

    X_train = (torch.tensor(X_train_np) - x_mean) / x_std
    y_train = (torch.tensor(y_train_np) - y_mean) / y_std

    # Pump work expected (analytical, from training data)
    W_pump_expected = torch.tensor(df_train[PUMP_COL].values.astype(np.float32))

    # -- Validation tensors -----------------------------------------------------
    X_val_np = df_val[INPUT_COLS].values.astype(np.float32)
    y_val_np = df_val[OUTPUT_COLS].values.astype(np.float32)
    X_val = (torch.tensor(X_val_np) - x_mean) / x_std
    y_val = (torch.tensor(y_val_np) - y_mean) / y_std

    # -- Collocation: fresh LHS over full input bounding box (A5) ---------------
    # These points have NO CoolProp labels - only physics residuals are evaluated.
    print("Generating LHS collocation points...")
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=8, seed=COLL_LHS_SEED)
    col_lhs = sampler.random(args.n_col).astype(np.float32)
    col_comp = _sample_compositions(col_lhs[:, :5])
    col_m = BOUNDS["m_dot"][0] + col_lhs[:, 5] * (BOUNDS["m_dot"][1] - BOUNDS["m_dot"][0])
    col_Ta = BOUNDS["T_amb"][0] + col_lhs[:, 6] * (BOUNDS["T_amb"][1] - BOUNDS["T_amb"][0])
    col_Ts = BOUNDS["T_sw"][0] + col_lhs[:, 7] * (BOUNDS["T_sw"][1] - BOUNDS["T_sw"][0])
    col_np = np.column_stack(
        [col_comp, col_m[:, None], col_Ta[:, None], col_Ts[:, None]]
    ).astype(np.float32)
    X_col = (torch.tensor(col_np) - x_mean) / x_std

    # Sanity-check: < 1% of collocation points should coincide with training data
    col_set = set(map(tuple, col_np[:, :6].round(6).tolist()))
    train_set = set(map(tuple, X_train_np[:, :6].round(6).tolist()))
    overlap = len(col_set & train_set) / len(col_set)
    print(f"Collocation-training overlap: {overlap:.3%} (should be < 1%)")

    # Pre-compute h_in / h_out for collocation points (needed for energy-balance residual).
    # Cache keyed by (n_col, COLL_LHS_SEED) — the LHS draws are deterministic from those.
    coll_cache = PROCESSED_DIR / f"collocation_h_n{args.n_col}_s{COLL_LHS_SEED}.parquet"
    if coll_cache.exists():
        try:
            cached = pd.read_parquet(coll_cache)
            if len(cached) == args.n_col and {"h_in", "h_out", "valid"} <= set(cached.columns):
                print(f"  Reusing cached collocation enthalpies from {coll_cache.name}")
                h_in_col_np = cached["h_in"].values.astype(np.float32)
                h_out_col_np = cached["h_out"].values.astype(np.float32)
                valid_mask = cached["valid"].values.astype(bool)
            else:
                print(f"  Cache {coll_cache.name} shape/columns mismatch, recomputing")
                h_in_col_np, h_out_col_np, valid_mask = _compute_h_in_out_col(col_np[:, :6])
        except Exception as exc:
            print(f"  Could not read {coll_cache.name}: {exc}; recomputing")
            h_in_col_np, h_out_col_np, valid_mask = _compute_h_in_out_col(col_np[:, :6])
    else:
        h_in_col_np, h_out_col_np, valid_mask = _compute_h_in_out_col(col_np[:, :6])
        pd.DataFrame(
            {"h_in": h_in_col_np, "h_out": h_out_col_np, "valid": valid_mask}
        ).to_parquet(coll_cache, index=False)
        print(f"  Cached collocation enthalpies to {coll_cache.name}")

    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        print(f"  Dropping {n_invalid} collocation points not liquid at storage conditions")
        col_np = col_np[valid_mask]
        X_col  = X_col[valid_mask]
        h_in_col_np  = h_in_col_np[valid_mask]
        h_out_col_np = h_out_col_np[valid_mask]
    h_in_col  = torch.tensor(h_in_col_np)
    h_out_col = torch.tensor(h_out_col_np)

    # -- Train ------------------------------------------------------------------
    model = train(
        X_train,
        y_train,
        X_col,
        h_in_col,
        h_out_col,
        W_pump_expected,
        scaler,
        X_val=X_val,
        y_val=y_val,
        n_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        lambda_energy=args.lambda_e,
        lambda_pump=args.lambda_p,
        patience=args.patience,
        resume=not args.no_resume,
        ckpt_every=args.ckpt_every,
    )
    print("Checkpoint saved to results/models/pinn_v1.pt")

    # -- Evaluate on held-out test set ------------------------------------------
    X_test_np = df_test[INPUT_COLS].values.astype(np.float32)
    y_test_np = df_test[OUTPUT_COLS].values.astype(np.float32)
    X_test = (torch.tensor(X_test_np) - x_mean) / x_std

    model.eval()
    with torch.no_grad():
        y_pred_test = scaler.unscale_y(model(X_test)).numpy()

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
    pd.DataFrame(eval_records).to_parquet(RESULTS_DIR / "surrogate_eval.parquet", index=False)
    print("Surrogate evaluation saved to results/tables/surrogate_eval.parquet")


if __name__ == "__main__":
    main()
