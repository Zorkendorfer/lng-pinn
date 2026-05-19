"""Run composition-aware dispatch and baselines over the full backtest."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.baseline import (
    COMP_COLS,
    optimize_blind_annual,
    optimize_blind_horizon,
    optimize_blind_lagged,
    optimize_constant_flow,
)
from lng_pinn.composition import CARGO_CYCLE_DAYS
from lng_pinn.dispatch import M_DOT_MAX, optimize
from lng_pinn.pinn import load

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")

CARGO_CYCLE_HOURS = CARGO_CYCLE_DAYS * 24  # 288 h — matches composition change cadence
CARGO_AMOUNT = 0.55  # fraction of TANK_CAP per delivery (~99 M kg, partial cargo)


def _append_records(
    records: list[dict[str, object]],
    window: pd.DataFrame,
    m_dot: object,
    cost_eur: object,
    n_hours: int,
) -> None:
    for t, row in enumerate(window.iloc[:n_hours].itertuples()):
        records.append(
            {
                "time": row.Index,
                "m_dot": float(m_dot[t]),  # type: ignore[index]
                "cost_eur": float(cost_eur[t]),  # type: ignore[index]
            }
        )


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
    step = 24

    annual_composition = ts[COMP_COLS].mean()
    starts = range(0, len(ts) - H + 1, step)
    n_windows = len(starts)
    aware_records: list[dict[str, object]] = []
    horizon_records: list[dict[str, object]] = []
    lagged_records: list[dict[str, object]] = []
    annual_records: list[dict[str, object]] = []
    constant_records: list[dict[str, object]] = []
    # Start at 85% so max frontloading (3.8%/day) can't drain to infeasibility before first cargo.
    inv_aware = inv_horizon = inv_lagged = inv_annual = inv_constant = 0.85

    demand_kg = M_DOT_MAX * 0.6 * H * 3600  # fixed; cargo schedule keeps tanks healthy

    for start in tqdm(starts, total=n_windows, desc="Dispatch windows", unit="day"):
        # Cargo delivery at cycle boundaries (same event for all strategies)
        if start > 0 and start % CARGO_CYCLE_HOURS == 0:
            inv_aware    = min(0.92, inv_aware    + CARGO_AMOUNT)
            inv_horizon  = min(0.92, inv_horizon  + CARGO_AMOUNT)
            inv_lagged   = min(0.92, inv_lagged   + CARGO_AMOUNT)
            inv_annual   = min(0.92, inv_annual   + CARGO_AMOUNT)
            inv_constant = min(0.92, inv_constant + CARGO_AMOUNT)

        window = ts.iloc[start : start + H]
        # Lagged composition: what the operator sees at the start of the window.
        lagged_composition = ts[COMP_COLS].iloc[start]
        record_hours = min(step, len(window))

        aware_sched = optimize(window, model, scaler, demand_kg, inv_aware)
        horizon_sched = optimize_blind_horizon(window, model, scaler, demand_kg, inv_horizon)
        lagged_sched = optimize_blind_lagged(
            window, model, scaler, demand_kg, lagged_composition, inv_lagged
        )
        annual_sched = optimize_blind_annual(
            window, model, scaler, demand_kg, annual_composition, inv_annual
        )
        constant_sched = optimize_constant_flow(window, model, scaler, demand_kg, inv_constant)

        _append_records(aware_records,    window, aware_sched.m_dot,    aware_sched.cost_eur,    record_hours)
        _append_records(horizon_records,  window, horizon_sched.m_dot,  horizon_sched.cost_eur,  record_hours)
        _append_records(lagged_records,   window, lagged_sched.m_dot,   lagged_sched.cost_eur,   record_hours)
        _append_records(annual_records,   window, annual_sched.m_dot,   annual_sched.cost_eur,   record_hours)
        _append_records(constant_records, window, constant_sched.m_dot, constant_sched.cost_eur, record_hours)

        inv_aware    = float(aware_sched.tank_level[record_hours])
        inv_horizon  = float(horizon_sched.tank_level[record_hours])
        inv_lagged   = float(lagged_sched.tank_level[record_hours])
        inv_annual   = float(annual_sched.tank_level[record_hours])
        inv_constant = float(constant_sched.tank_level[record_hours])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aware_df    = pd.DataFrame(aware_records)
    horizon_df  = pd.DataFrame(horizon_records)
    lagged_df   = pd.DataFrame(lagged_records)
    annual_df   = pd.DataFrame(annual_records)
    constant_df = pd.DataFrame(constant_records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aware_df.to_parquet(RESULTS_DIR / "dispatch_v1.parquet", index=False)
    horizon_df.to_parquet(RESULTS_DIR / "baseline_horizon_v1.parquet", index=False)
    lagged_df.to_parquet(RESULTS_DIR / "baseline_lagged_v1.parquet", index=False)
    annual_df.to_parquet(RESULTS_DIR / "baseline_annual_v1.parquet", index=False)
    constant_df.to_parquet(RESULTS_DIR / "baseline_constant_v1.parquet", index=False)
    horizon_df.to_parquet(RESULTS_DIR / "baseline_v1.parquet", index=False)

    def _pct(baseline: float, aware: float) -> str:
        return f"{(baseline - aware) / baseline * 100:.2f}%"

    total_aware    = float(aware_df["cost_eur"].sum())
    total_horizon  = float(horizon_df["cost_eur"].sum())
    total_lagged   = float(lagged_df["cost_eur"].sum())
    total_annual   = float(annual_df["cost_eur"].sum())
    total_constant = float(constant_df["cost_eur"].sum())
    print(f"Total aware cost:          {total_aware:>13,.0f} EUR")
    print(f"Total blind-lagged cost:   {total_lagged:>13,.0f} EUR  saving={_pct(total_lagged, total_aware)}")
    print(f"Total blind-horizon cost:  {total_horizon:>13,.0f} EUR  saving={_pct(total_horizon, total_aware)}")
    print(f"Total blind-annual cost:   {total_annual:>13,.0f} EUR  saving={_pct(total_annual, total_aware)}")
    print(f"Total constant-flow cost:  {total_constant:>13,.0f} EUR  saving={_pct(total_constant, total_aware)}")


if __name__ == "__main__":
    main()
