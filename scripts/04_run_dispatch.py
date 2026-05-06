"""Run composition-aware and composition-blind dispatch over the full backtest."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.baseline import optimize_blind
from lng_pinn.dispatch import M_DOT_MAX, optimize
from lng_pinn.pinn import load

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results/tables")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-days", type=int, default=7)
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}  horizon_days={args.horizon_days}")

    model, scaler = load()
    model.eval()

    ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts.index = pd.to_datetime(ts.index, utc=True)

    H = args.horizon_days * 24
    step = H  # non-overlapping rolling windows

    aware_records, blind_records = [], []
    for start in range(0, len(ts) - H, step):
        window = ts.iloc[start : start + H]
        demand_kg = M_DOT_MAX * 0.6 * H * 3600  # 60% utilisation target

        aware_sched = optimize(window, model, scaler, demand_kg)
        blind_sched = optimize_blind(window, model, scaler, demand_kg)

        for t, row in enumerate(window.itertuples()):
            aware_records.append({"time": row.Index, "m_dot": aware_sched.m_dot[t],
                                   "cost_eur": aware_sched.cost_eur[t]})
            blind_records.append({"time": row.Index, "m_dot": blind_sched.m_dot[t],
                                   "cost_eur": blind_sched.cost_eur[t]})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aware_records).to_parquet(RESULTS_DIR / "dispatch_v1.parquet", index=False)
    pd.DataFrame(blind_records).to_parquet(RESULTS_DIR / "baseline_v1.parquet", index=False)

    total_aware = sum(r["cost_eur"] for r in aware_records)
    total_blind = sum(r["cost_eur"] for r in blind_records)
    saving_pct = (total_blind - total_aware) / total_blind * 100
    print(f"Total aware cost: {total_aware:,.0f} EUR")
    print(f"Total blind cost: {total_blind:,.0f} EUR")
    print(f"Saving: {saving_pct:.2f}%")


if __name__ == "__main__":
    main()
