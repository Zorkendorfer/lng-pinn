"""Composition seed sensitivity analysis.

Re-runs the rolling-horizon dispatch backtest with 5 different composition seeds.
Reports mean ± std of yearly saving (aware vs blind-horizon) across seeds.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.baseline import COMP_COLS, optimize_blind_horizon, optimize_blind_lagged
from lng_pinn.composition import CARGO_CYCLE_DAYS, build_composition_series
from lng_pinn.dispatch import M_DOT_MAX, optimize
from lng_pinn.pinn import load

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
SEEDS = [42, 0, 1, 7, 13]
HORIZON_DAYS = 7
CARGO_CYCLE_HOURS = CARGO_CYCLE_DAYS * 24
CARGO_AMOUNT = 0.55
STRATEGIES = ("aware", "lagged", "horizon")


def _seed_result_path(seed: int) -> Path:
    """Final per-seed backtest result (consolidated parquet)."""
    return PROCESSED_DIR / f"seed_sensitivity_seed{seed}.parquet"


def _seed_partial_records_path(seed: int) -> Path:
    return PROCESSED_DIR / f"seed_sensitivity_partial_seed{seed}.parquet"


def _seed_partial_state_path(seed: int) -> Path:
    return PROCESSED_DIR / f"seed_sensitivity_partial_seed{seed}.json"


def _ts_for_seed(seed: int) -> pd.DataFrame:
    """Swap composition columns in the cached timeseries for the given seed."""
    ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts.index = pd.to_datetime(ts.index, utc=True)
    comp = build_composition_series(ts.index, seed=seed)
    for col in COMP_COLS:
        ts[col] = comp[col]
    return ts


def _save_seed_partial(
    records_by_strategy: dict[str, list[dict[str, object]]],
    inv: dict[str, float],
    next_start: int,
    seed: int,
) -> None:
    """Atomic flush of per-seed intra-backtest state."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for strategy, records in records_by_strategy.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        df["_strategy"] = strategy
        frames.append(df)
    rec_path = _seed_partial_records_path(seed)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        tmp = rec_path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(rec_path)
    state = {"next_start": next_start, "inv": inv}
    state_path = _seed_partial_state_path(seed)
    tmp_state = state_path.with_suffix(".json.tmp")
    tmp_state.write_text(json.dumps(state))
    tmp_state.replace(state_path)


def _load_seed_partial(
    seed: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, float], int] | None:
    state_path = _seed_partial_state_path(seed)
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
        next_start = int(state["next_start"])
        inv = {k: float(v) for k, v in state["inv"].items()}
        records: dict[str, list[dict[str, object]]] = {s: [] for s in STRATEGIES}
        rec_path = _seed_partial_records_path(seed)
        if rec_path.exists():
            combined = pd.read_parquet(rec_path)
            for strategy, group in combined.groupby("_strategy", sort=False):
                records[str(strategy)] = group.drop(
                    columns=["_strategy"]
                ).to_dict(orient="records")
        return records, inv, next_start
    except Exception as exc:
        print(f"  seed={seed}: could not load partial ({exc}); restarting backtest")
        return None


def _clear_seed_partial(seed: int) -> None:
    for path in (_seed_partial_state_path(seed), _seed_partial_records_path(seed)):
        if path.exists():
            path.unlink()


def _run_backtest(
    ts: pd.DataFrame,
    model: object,
    scaler: object,
    seed: int,
    resume: bool = True,
    ckpt_every: int = 20,
    carbon_price_eur_per_t: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    H = HORIZON_DAYS * 24
    step = 24
    starts = list(range(0, len(ts) - H + 1, step))
    demand_kg = M_DOT_MAX * 0.6 * H * 3600

    records_by_strategy: dict[str, list[dict[str, object]]] = {s: [] for s in STRATEGIES}
    inv = {s: 0.85 for s in STRATEGIES}
    resume_start = 0

    if resume:
        loaded = _load_seed_partial(seed)
        if loaded is not None:
            records_by_strategy, inv, resume_start = loaded
            n_recs = next(iter(records_by_strategy.values()), [])
            print(
                f"  seed={seed}: resuming with {len(n_recs)} hours/strategy cached; "
                f"next start={resume_start}"
            )

    pending_starts = [s for s in starts if s >= resume_start]
    flush_counter = 0
    last_completed = resume_start

    pbar = tqdm(pending_starts, total=len(pending_starts), desc="  windows", leave=False)
    for start in pbar:
        if start > 0 and start % CARGO_CYCLE_HOURS == 0:
            for s in STRATEGIES:
                inv[s] = min(0.92, inv[s] + CARGO_AMOUNT)

        window = ts.iloc[start : start + H]
        lagged_composition = ts[COMP_COLS].iloc[start]
        n = min(step, len(window))

        cp = carbon_price_eur_per_t
        a_sched = optimize(  # type: ignore[arg-type]
            window, model, scaler, demand_kg, inv["aware"], carbon_price_eur_per_t=cp,
        )
        l_sched = optimize_blind_lagged(  # type: ignore[arg-type]
            window, model, scaler, demand_kg, lagged_composition, inv["lagged"],
            carbon_price_eur_per_t=cp,
        )
        h_sched = optimize_blind_horizon(  # type: ignore[arg-type]
            window, model, scaler, demand_kg, inv["horizon"], carbon_price_eur_per_t=cp,
        )

        for t, row in enumerate(window.iloc[:n].itertuples()):
            records_by_strategy["aware"].append(
                {"time": row.Index, "cost_eur": float(a_sched.cost_eur[t])}
            )
            records_by_strategy["lagged"].append(
                {"time": row.Index, "cost_eur": float(l_sched.cost_eur[t])}
            )
            records_by_strategy["horizon"].append(
                {"time": row.Index, "cost_eur": float(h_sched.cost_eur[t])}
            )

        inv["aware"]   = float(a_sched.tank_level[n])
        inv["lagged"]  = float(l_sched.tank_level[n])
        inv["horizon"] = float(h_sched.tank_level[n])

        last_completed = start + step
        flush_counter += 1
        if flush_counter >= ckpt_every:
            _save_seed_partial(records_by_strategy, inv, last_completed, seed)
            flush_counter = 0

    if pending_starts:
        _save_seed_partial(records_by_strategy, inv, last_completed, seed)

    aware_df   = pd.DataFrame(records_by_strategy["aware"]).set_index("time")
    lagged_df  = pd.DataFrame(records_by_strategy["lagged"]).set_index("time")
    horizon_df = pd.DataFrame(records_by_strategy["horizon"]).set_index("time")
    return aware_df, lagged_df, horizon_df


def _save_seed_result(
    aware_df: pd.DataFrame,
    lagged_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    seed: int,
) -> None:
    """Persist the consolidated per-seed backtest so future runs can skip it."""
    parts = []
    for name, df in [("aware", aware_df), ("lagged", lagged_df), ("horizon", horizon_df)]:
        d = df.reset_index()
        d["_strategy"] = name
        parts.append(d)
    combined = pd.concat(parts, ignore_index=True)
    path = _seed_result_path(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)


def _load_seed_result(
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    path = _seed_result_path(seed)
    if not path.exists():
        return None
    try:
        combined = pd.read_parquet(path)
        combined["time"] = pd.to_datetime(combined["time"], utc=True)
        dfs = {}
        for name in STRATEGIES:
            d = combined[combined["_strategy"] == name].drop(columns=["_strategy"])
            dfs[name] = d.set_index("time")
        return dfs["aware"], dfs["lagged"], dfs["horizon"]
    except Exception as exc:
        print(f"  seed={seed}: could not load cached result ({exc}); recomputing")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=20,
        help="Flush per-seed partial state every K windows (default 20).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore cached per-seed results and partials; rerun all seeds from scratch.",
    )
    parser.add_argument(
        "--carbon-price", type=float, default=0.0,
        help="v1.3 B1 CO2 price in EUR per tonne. Cached seed results are invalidated "
             "whenever this changes value across runs.",
    )
    args = parser.parse_args()
    resume = not args.no_resume

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}  seeds={SEEDS}  carbon_price={args.carbon_price:.1f} EUR/tCO2")

    model, scaler = load()
    model.eval()

    all_records: list[dict] = []

    for seed in tqdm(SEEDS, desc="Seeds"):
        # Skip the whole backtest if we have a cached consolidated result.
        cached = _load_seed_result(seed) if resume else None
        if cached is not None:
            aware_df, lagged_df, horizon_df = cached
            print(f"  seed={seed}: reusing cached backtest result")
        else:
            ts = _ts_for_seed(seed)
            aware_df, lagged_df, horizon_df = _run_backtest(
                ts, model, scaler, seed=seed, resume=resume, ckpt_every=args.ckpt_every,
                carbon_price_eur_per_t=args.carbon_price,
            )
            _save_seed_result(aware_df, lagged_df, horizon_df, seed)
            _clear_seed_partial(seed)

        yearly_aware   = aware_df["cost_eur"].resample("YE").sum()
        yearly_lagged  = lagged_df["cost_eur"].resample("YE").sum()
        yearly_horizon = horizon_df["cost_eur"].resample("YE").sum()

        for ts_end in yearly_aware.index:
            all_records.append({
                "seed": seed,
                "year": ts_end.year,
                "aware_eur":          float(yearly_aware[ts_end]),
                "blind_lagged_eur":   float(yearly_lagged[ts_end]),
                "blind_horizon_eur":  float(yearly_horizon[ts_end]),
                "saving_vs_lagged_pct": float(
                    (yearly_lagged[ts_end] - yearly_aware[ts_end]) / yearly_lagged[ts_end] * 100
                ),
                "saving_vs_horizon_pct": float(
                    (yearly_horizon[ts_end] - yearly_aware[ts_end]) / yearly_horizon[ts_end] * 100
                ),
            })

        total_lagged_pct = (
            (lagged_df["cost_eur"].sum() - aware_df["cost_eur"].sum())
            / lagged_df["cost_eur"].sum() * 100
        )
        total_horizon_pct = (
            (horizon_df["cost_eur"].sum() - aware_df["cost_eur"].sum())
            / horizon_df["cost_eur"].sum() * 100
        )
        print(f"  seed={seed}  lagged={total_lagged_pct:.2f}%  horizon={total_horizon_pct:.2f}%")

    results_df = pd.DataFrame(all_records)

    # Per-year mean ± std across seeds, for both baselines
    summary_rows = []
    for baseline in ["lagged", "horizon"]:
        col = f"saving_vs_{baseline}_pct"
        grp = results_df.groupby("year")[col].agg(mean="mean", std="std").reset_index()
        grp["baseline"] = baseline
        summary_rows.append(grp)
    summary = pd.concat(summary_rows, ignore_index=True)

    # Seed-averaged overall saving
    seed_totals = results_df.groupby("seed")["saving_vs_lagged_pct"].mean()
    overall_mean = float(seed_totals.mean())
    overall_std = float(seed_totals.std())
    n = len(SEEDS)
    print(f"\nOverall saving vs lagged:  {overall_mean:.2f}% ± {overall_std:.2f}%  (n={n} seeds)")
    h_totals = results_df.groupby("seed")["saving_vs_horizon_pct"].mean()
    print(f"Overall saving vs horizon: {h_totals.mean():.2f}% ± {h_totals.std():.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "seed_sensitivity.csv", index=False)
    summary.to_csv(RESULTS_DIR / "seed_sensitivity_summary.csv", index=False)
    print("Saved seed_sensitivity.csv and seed_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
