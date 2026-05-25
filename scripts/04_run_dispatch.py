"""Run composition-aware dispatch and baselines over the full backtest."""

import argparse
import json
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

STRATEGIES = ("aware", "horizon", "lagged", "annual", "constant")
INV_INITIAL = 0.85  # see comment in main(): high enough to survive frontloading before first cargo

PARTIAL_RECORDS = PROCESSED_DIR / "dispatch_partial.parquet"
PARTIAL_STATE = PROCESSED_DIR / "dispatch_partial_state.json"

FINAL_OUTPUTS = {
    "aware":    "dispatch_v1.parquet",
    "horizon":  "baseline_horizon_v1.parquet",
    "lagged":   "baseline_lagged_v1.parquet",
    "annual":   "baseline_annual_v1.parquet",
    "constant": "baseline_constant_v1.parquet",
}


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


def _save_partial(
    records_by_strategy: dict[str, list[dict[str, object]]],
    inv: dict[str, float],
    next_start: int,
) -> None:
    """Flush an atomic partial-state snapshot to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for strategy, records in records_by_strategy.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        df["_strategy"] = strategy
        frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        tmp = PARTIAL_RECORDS.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(PARTIAL_RECORDS)
    state = {"next_start": next_start, "inv": inv}
    tmp_state = PARTIAL_STATE.with_suffix(".json.tmp")
    tmp_state.write_text(json.dumps(state))
    tmp_state.replace(PARTIAL_STATE)


def _load_partial() -> (
    tuple[dict[str, list[dict[str, object]]], dict[str, float], int] | None
):
    """Reload partial state; returns (records_by_strategy, inv, next_start) or None."""
    if not PARTIAL_STATE.exists():
        return None
    try:
        state = json.loads(PARTIAL_STATE.read_text())
        next_start = int(state["next_start"])
        inv = {k: float(v) for k, v in state["inv"].items()}
        records_by_strategy: dict[str, list[dict[str, object]]] = {s: [] for s in STRATEGIES}
        if PARTIAL_RECORDS.exists():
            combined = pd.read_parquet(PARTIAL_RECORDS)
            for strategy, group in combined.groupby("_strategy", sort=False):
                records_by_strategy[str(strategy)] = group.drop(
                    columns=["_strategy"]
                ).to_dict(orient="records")
        return records_by_strategy, inv, next_start
    except Exception as exc:
        print(f"  Could not load dispatch partial: {exc}; starting fresh")
        return None


def _clear_partial() -> None:
    for path in (PARTIAL_STATE, PARTIAL_RECORDS):
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=20,
        help="Flush partial state every K dispatch windows (default 20).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing dispatch_partial.* and start from the first window.",
    )
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
    starts = list(range(0, len(ts) - H + 1, step))
    n_windows = len(starts)
    demand_kg = M_DOT_MAX * 0.6 * H * 3600  # fixed; cargo schedule keeps tanks healthy

    # --- Initialise state, then try to resume -----------------------------------
    records_by_strategy: dict[str, list[dict[str, object]]] = {s: [] for s in STRATEGIES}
    inv: dict[str, float] = {s: INV_INITIAL for s in STRATEGIES}
    resume_start_value = 0

    if not args.no_resume:
        loaded = _load_partial()
        if loaded is not None:
            records_by_strategy, inv, resume_start_value = loaded
            n_recs = next(iter(records_by_strategy.values()), [])
            print(
                f"  Resumed: {len(n_recs)} hours per strategy already cached; "
                f"next window start={resume_start_value}"
            )

    pending_starts = [s for s in starts if s >= resume_start_value]
    if not pending_starts:
        print("  All dispatch windows already complete; writing finals.")

    flush_counter = 0
    last_completed = resume_start_value
    pbar = tqdm(pending_starts, total=len(pending_starts), desc="Dispatch windows", unit="day")
    for start in pbar:
        # Cargo delivery at cycle boundaries (deterministic from `start`, so resume-safe).
        if start > 0 and start % CARGO_CYCLE_HOURS == 0:
            for s in STRATEGIES:
                inv[s] = min(0.92, inv[s] + CARGO_AMOUNT)

        window = ts.iloc[start : start + H]
        lagged_composition = ts[COMP_COLS].iloc[start]
        record_hours = min(step, len(window))

        aware_sched    = optimize(window, model, scaler, demand_kg, inv["aware"])
        horizon_sched  = optimize_blind_horizon(window, model, scaler, demand_kg, inv["horizon"])
        lagged_sched   = optimize_blind_lagged(
            window, model, scaler, demand_kg, lagged_composition, inv["lagged"]
        )
        annual_sched   = optimize_blind_annual(
            window, model, scaler, demand_kg, annual_composition, inv["annual"]
        )
        constant_sched = optimize_constant_flow(
            window, model, scaler, demand_kg, inv["constant"]
        )

        _append_records(
            records_by_strategy["aware"], window,
            aware_sched.m_dot, aware_sched.cost_eur, record_hours,
        )
        _append_records(
            records_by_strategy["horizon"], window,
            horizon_sched.m_dot, horizon_sched.cost_eur, record_hours,
        )
        _append_records(
            records_by_strategy["lagged"], window,
            lagged_sched.m_dot, lagged_sched.cost_eur, record_hours,
        )
        _append_records(
            records_by_strategy["annual"], window,
            annual_sched.m_dot, annual_sched.cost_eur, record_hours,
        )
        _append_records(
            records_by_strategy["constant"], window,
            constant_sched.m_dot, constant_sched.cost_eur, record_hours,
        )

        inv["aware"]    = float(aware_sched.tank_level[record_hours])
        inv["horizon"]  = float(horizon_sched.tank_level[record_hours])
        inv["lagged"]   = float(lagged_sched.tank_level[record_hours])
        inv["annual"]   = float(annual_sched.tank_level[record_hours])
        inv["constant"] = float(constant_sched.tank_level[record_hours])

        last_completed = start + step
        flush_counter += 1
        if flush_counter >= args.ckpt_every:
            _save_partial(records_by_strategy, inv, last_completed)
            flush_counter = 0

    # Final partial flush so the last batch isn't lost if writes below fail.
    if pending_starts:
        _save_partial(records_by_strategy, inv, last_completed)

    # --- Write final per-strategy parquets --------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_dfs: dict[str, pd.DataFrame] = {}
    for strategy, fname in FINAL_OUTPUTS.items():
        df = pd.DataFrame(records_by_strategy[strategy])
        df.to_parquet(RESULTS_DIR / fname, index=False)
        final_dfs[strategy] = df
    # Keep legacy alias used by some downstream scripts/notebooks.
    final_dfs["horizon"].to_parquet(RESULTS_DIR / "baseline_v1.parquet", index=False)

    _clear_partial()

    def _pct(baseline: float, aware: float) -> str:
        return f"{(baseline - aware) / baseline * 100:.2f}%"

    total_aware    = float(final_dfs["aware"]["cost_eur"].sum())
    total_horizon  = float(final_dfs["horizon"]["cost_eur"].sum())
    total_lagged   = float(final_dfs["lagged"]["cost_eur"].sum())
    total_annual   = float(final_dfs["annual"]["cost_eur"].sum())
    total_constant = float(final_dfs["constant"]["cost_eur"].sum())
    print(f"Total aware cost:         {total_aware:>13,.0f} EUR")
    print(f"Total blind-lagged:   {total_lagged:>13,.0f} EUR  {_pct(total_lagged, total_aware)}")
    print(f"Total blind-horizon:  {total_horizon:>13,.0f} EUR  {_pct(total_horizon, total_aware)}")
    print(f"Total blind-annual:   {total_annual:>13,.0f} EUR  {_pct(total_annual, total_aware)}")
    pct_const = _pct(total_constant, total_aware)
    print(f"Total constant-flow:  {total_constant:>13,.0f} EUR  {pct_const}")

    # n_windows kept for parity with earlier prints if needed downstream.
    _ = n_windows


if __name__ == "__main__":
    main()
