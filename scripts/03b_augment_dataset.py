"""v1.3 A2: augment the training set with CoolProp-labelled dispatch-trajectory rows.

Workflow:
  1. Load the v1.3 trained PINN + the full timeseries.
  2. Run the *aware* dispatch strategy over the whole horizon to produce
     the trajectory the model is actually queried at in production.
  3. Extract unique (composition, m_dot, T_amb, T_sw) operating points
     from the trajectory, rounded to 4 dp to dedupe near-duplicates.
  4. Label each unique point through CoolProp in parallel (reusing
     dataset._simulate_one).
  5. Append the new rows to data/processed/train.parquet with
     _source="trajectory". Rows already in the training set are skipped.
  6. Print a summary; the user reruns 03_train_pinn.py --no-resume on
     the augmented set to get the trajectory-aware model.

Run this *after* A1 (relative-cost loss) is trained in. Augmenting on an
uncorrected trajectory would reinforce the cost bias rather than fix it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.composition import CARGO_CYCLE_DAYS
from lng_pinn.dataset import _simulate_one, append_trajectory_rows
from lng_pinn.dispatch import M_DOT_MAX, optimize
from lng_pinn.pinn import load

PROCESSED_DIR = Path("data/processed")
COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]
HORIZON_DAYS = 7
INV_INITIAL = 0.85
CARGO_AMOUNT = 0.55


def _aware_trajectory(
    ts: pd.DataFrame, model, scaler, carbon_price: float
) -> pd.DataFrame:
    """Run the aware strategy across the full timeseries, return per-hour records."""
    H = HORIZON_DAYS * 24
    step = 24
    cargo_cycle_hours = CARGO_CYCLE_DAYS * 24
    starts = list(range(0, len(ts) - H + 1, step))
    demand_kg = M_DOT_MAX * 0.6 * H * 3600
    inv = INV_INITIAL

    records: list[dict] = []
    for start in tqdm(starts, desc="Aware dispatch", unit="day"):
        if start > 0 and start % cargo_cycle_hours == 0:
            inv = min(0.92, inv + CARGO_AMOUNT)
        window = ts.iloc[start : start + H]
        sched = optimize(
            window, model, scaler, demand_kg, inv,
            carbon_price_eur_per_t=carbon_price,
        )
        n = min(step, len(window))
        for t, row in enumerate(window.iloc[:n].itertuples()):
            records.append({
                "time": row.Index,
                "m_dot": float(sched.m_dot[t]),
                **{c: float(getattr(row, c)) for c in COMP_COLS},
                "T_amb": float(row.T_amb),
                "T_sw": float(row.T_sw),
            })
        inv = float(sched.tank_level[n])
    return pd.DataFrame(records)


def _unique_operating_points(traj: pd.DataFrame, decimals: int = 4) -> list[tuple]:
    """Round and dedupe trajectory rows into unique (comp, m_dot, T_amb, T_sw) tuples."""
    key_cols = COMP_COLS + ["m_dot", "T_amb", "T_sw"]
    rounded = traj[key_cols].round(decimals)
    unique = rounded.drop_duplicates().reset_index(drop=True)
    return [
        (
            tuple(float(unique.iloc[i][c]) for c in COMP_COLS),
            float(unique.iloc[i]["m_dot"]),
            float(unique.iloc[i]["T_amb"]),
            float(unique.iloc[i]["T_sw"]),
        )
        for i in range(len(unique))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--carbon-price", type=float, default=0.0,
        help="CO2 price used when generating the trajectory. Use the same value "
             "you plan to evaluate at downstream, since dispatch choices shift "
             "with the carbon term.",
    )
    parser.add_argument(
        "--dedupe-decimals", type=int, default=4,
        help="Decimal places to round m_dot/T to when deduping trajectory points.",
    )
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}  carbon_price={args.carbon_price:.1f} EUR/tCO2")

    model, scaler = load()
    model.eval()
    ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts.index = pd.to_datetime(ts.index, utc=True)

    traj = _aware_trajectory(ts, model, scaler, args.carbon_price)
    print(f"Trajectory: {len(traj)} hours")

    points = _unique_operating_points(traj, decimals=args.dedupe_decimals)
    print(f"Unique operating points after dedupe: {len(points)}")

    n_workers = max(1, os.cpu_count() or 1)
    labelled: list[dict] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_simulate_one, p) for p in points]
        for f in tqdm(as_completed(futures), total=len(futures), desc="CoolProp label", unit="pt"):
            r = f.result()
            if r is None:
                failures += 1
            else:
                labelled.append(r)
    print(f"  Labelled: {len(labelled)}   CoolProp failures: {failures}")

    n_added = append_trajectory_rows(labelled)
    print(f"  Appended {n_added} new rows to data/processed/train.parquet")
    skipped = len(labelled) - n_added
    if skipped:
        print(f"  Skipped {skipped} rows already present in the LHS training set")

    print()
    print("Next step: retrain on the augmented dataset:")
    print("  uv run python scripts/03_train_pinn.py --no-resume")


if __name__ == "__main__":
    main()
