"""Composition seed sensitivity analysis.

Re-runs the rolling-horizon dispatch backtest with 5 different composition seeds.
Reports mean ± std of yearly saving (aware vs blind-horizon) across seeds.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.baseline import COMP_COLS, optimize_blind_horizon
from lng_pinn.composition import CARGO_CYCLE_DAYS, build_composition_series
from lng_pinn.dispatch import M_DOT_MAX, optimize
from lng_pinn.pinn import load

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
SEEDS = [42, 0, 1, 7, 13]
HORIZON_DAYS = 7
CARGO_CYCLE_HOURS = CARGO_CYCLE_DAYS * 24
CARGO_AMOUNT = 0.55


def _ts_for_seed(seed: int) -> pd.DataFrame:
    """Swap composition columns in the cached timeseries for the given seed."""
    ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts.index = pd.to_datetime(ts.index, utc=True)
    comp = build_composition_series(ts.index, seed=seed)
    for col in COMP_COLS:
        ts[col] = comp[col]
    return ts


def _run_backtest(
    ts: pd.DataFrame,
    model: object,
    scaler: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    H = HORIZON_DAYS * 24
    step = 24
    starts = range(0, len(ts) - H + 1, step)
    aware_records: list[dict] = []
    horizon_records: list[dict] = []
    inv_aware = inv_horizon = 0.85
    demand_kg = M_DOT_MAX * 0.6 * H * 3600

    for start in tqdm(starts, desc=f"  windows", leave=False):
        if start > 0 and start % CARGO_CYCLE_HOURS == 0:
            inv_aware   = min(0.92, inv_aware   + CARGO_AMOUNT)
            inv_horizon = min(0.92, inv_horizon + CARGO_AMOUNT)

        window = ts.iloc[start : start + H]
        n = min(step, len(window))

        a_sched = optimize(window, model, scaler, demand_kg, inv_aware)  # type: ignore[arg-type]
        h_sched = optimize_blind_horizon(window, model, scaler, demand_kg, inv_horizon)  # type: ignore[arg-type]

        for t, row in enumerate(window.iloc[:n].itertuples()):
            aware_records.append({"time": row.Index, "cost_eur": float(a_sched.cost_eur[t])})
            horizon_records.append({"time": row.Index, "cost_eur": float(h_sched.cost_eur[t])})

        inv_aware = float(a_sched.tank_level[n])
        inv_horizon = float(h_sched.tank_level[n])

    aware_df = pd.DataFrame(aware_records).set_index("time")
    horizon_df = pd.DataFrame(horizon_records).set_index("time")
    return aware_df, horizon_df


def main() -> None:
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}  seeds={SEEDS}")

    model, scaler = load()
    model.eval()

    all_records: list[dict] = []

    for seed in tqdm(SEEDS, desc="Seeds"):
        ts = _ts_for_seed(seed)
        aware_df, horizon_df = _run_backtest(ts, model, scaler)

        yearly_aware = aware_df["cost_eur"].resample("YE").sum()
        yearly_horizon = horizon_df["cost_eur"].resample("YE").sum()
        saving_pct = (yearly_horizon - yearly_aware) / yearly_horizon * 100

        for ts_end, pct in saving_pct.items():
            all_records.append(
                {
                    "seed": seed,
                    "year": ts_end.year,
                    "aware_eur": float(yearly_aware[ts_end]),
                    "blind_horizon_eur": float(yearly_horizon[ts_end]),
                    "saving_pct": float(pct),
                }
            )

        total_pct = (
            (horizon_df["cost_eur"].sum() - aware_df["cost_eur"].sum())
            / horizon_df["cost_eur"].sum()
            * 100
        )
        print(f"  seed={seed}  total saving={total_pct:.2f}%")

    results_df = pd.DataFrame(all_records)

    # Per-year mean ± std across seeds
    summary = (
        results_df.groupby("year")["saving_pct"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    summary.columns = ["year", "saving_pct_mean", "saving_pct_std"]

    # Seed-averaged overall saving
    seed_totals = results_df.groupby("seed")["saving_pct"].mean()
    overall_mean = float(seed_totals.mean())
    overall_std = float(seed_totals.std())
    print(
        f"\nOverall saving: {overall_mean:.2f}% ± {overall_std:.2f}%  (n={len(SEEDS)} seeds)"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "seed_sensitivity.csv", index=False)
    summary.to_csv(RESULTS_DIR / "seed_sensitivity_summary.csv", index=False)
    print("Saved seed_sensitivity.csv and seed_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
