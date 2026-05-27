"""Composition seed sensitivity analysis.

Re-runs the rolling-horizon dispatch backtest with 5 different composition seeds.
Reports mean ± std of yearly saving (aware vs blind-horizon) across seeds.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
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


def _seed_true_cost_done_path(seed: int, strategy: str) -> Path:
    """Completed per-(seed, strategy) CoolProp true-cost cache."""
    return PROCESSED_DIR / f"seed_true_costs_seed{seed}_{strategy}.parquet"


def _seed_true_cost_partial_path(seed: int, strategy: str) -> Path:
    """In-progress per-row true-cost partial, flushed every K completions."""
    return PROCESSED_DIR / f"seed_true_costs_seed{seed}_{strategy}_inprogress.parquet"


def _flush_true_cost_partial(done: dict[int, float], path: Path) -> None:
    """Atomic write of the (_row_idx, true_cost_eur) dict to a partial parquet."""
    if not done:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"_row_idx": list(done.keys()), "true_cost_eur": list(done.values())}
    )
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _true_cost_row(args: tuple) -> float | None:
    """Worker: one CoolProp simulation + cost calculation (EUR/h).

    Top-level so ProcessPoolExecutor can pickle it. Locally imports plant /
    thermo so worker startup doesn't pay for the parent's full import graph.
    Mirrors the equivalent helper in scripts/07_carbon_sweep.py.
    """
    composition, m_dot, T_amb, T_sw, price, carbon_price = args
    from lng_pinn.plant import simulate
    from lng_pinn.thermo import co2_per_kg_fuel

    try:
        res = simulate(composition, m_dot, T_amb, T_sw)
    except ValueError:
        return None
    elec = price * res.W_total * m_dot * 3.6
    if carbon_price > 0.0:
        carbon = carbon_price * co2_per_kg_fuel(composition) * m_dot * 3.6
    else:
        carbon = 0.0
    return float(elec + carbon)


def _eval_true_cost_for_seed_strategy(
    dispatch_df: pd.DataFrame,
    ts: pd.DataFrame,
    seed: int,
    strategy: str,
    carbon_price: float,
    resume: bool = True,
    ckpt_every: int = 2000,
) -> pd.Series:
    """CoolProp ground-truth cost (EUR/h) per hour for one (seed, strategy).

    Parallel across processes (one CoolProp call per row). Flushes a per-row
    partial every ``ckpt_every`` completions to
    seed_true_costs_seed<seed>_<strategy>_inprogress.parquet so a Ctrl-C only
    loses the rows since the last flush.
    """
    joined = dispatch_df.join(ts, how="inner")
    n = len(joined)
    arg_list = [
        (
            tuple(float(getattr(row, c)) for c in COMP_COLS),
            float(row.m_dot),
            float(row.T_amb),
            float(row.T_sw),
            float(row.price_eur_mwh),
            float(carbon_price),
        )
        for row in joined.itertuples()
    ]

    partial_path = _seed_true_cost_partial_path(seed, strategy)
    done: dict[int, float] = {}
    if resume and partial_path.exists():
        try:
            prior = pd.read_parquet(partial_path)
            for r in prior.itertuples(index=False):
                done[int(r._row_idx)] = float(r.true_cost_eur)
        except Exception:
            done = {}

    pending = [(i, a) for i, a in enumerate(arg_list) if i not in done]
    if pending:
        n_workers = max(1, os.cpu_count() or 1)
        completed_since_ckpt = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_true_cost_row, a): i for i, a in pending}
            for future in tqdm(
                as_completed(futures), total=len(pending),
                desc=f"  true-cost seed={seed} {strategy}", unit="hr", leave=False,
            ):
                i = futures[future]
                cost = future.result()
                if cost is not None:
                    done[i] = cost
                completed_since_ckpt += 1
                if completed_since_ckpt >= ckpt_every:
                    _flush_true_cost_partial(done, partial_path)
                    completed_since_ckpt = 0
        _flush_true_cost_partial(done, partial_path)

    out = [done.get(i, np.nan) for i in range(n)]
    return pd.Series(out, index=joined.index, name="true_cost_eur")


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
            # m_dot is persisted alongside cost_eur so the Phase-2 CoolProp
            # re-evaluation can rebuild the (composition, m_dot, weather)
            # tuple from the cached dispatch record + the per-seed timeseries.
            records_by_strategy["aware"].append({
                "time": row.Index,
                "m_dot": float(a_sched.m_dot[t]),
                "cost_eur": float(a_sched.cost_eur[t]),
            })
            records_by_strategy["lagged"].append({
                "time": row.Index,
                "m_dot": float(l_sched.m_dot[t]),
                "cost_eur": float(l_sched.cost_eur[t]),
            })
            records_by_strategy["horizon"].append({
                "time": row.Index,
                "m_dot": float(h_sched.m_dot[t]),
                "cost_eur": float(h_sched.cost_eur[t]),
            })

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
        if "m_dot" not in combined.columns:
            # Pre-Phase-2 cache: missing m_dot means we cannot run the
            # CoolProp true-cost re-eval. Force a recompute.
            print(
                f"  seed={seed}: cached result is missing m_dot (pre-Phase-2); recomputing"
            )
            return None
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
        # ----- Phase 1: dispatch backtest ---------------------------------------
        cached = _load_seed_result(seed) if resume else None
        if cached is not None:
            aware_df, lagged_df, horizon_df = cached
            print(f"  seed={seed}: reusing cached backtest result")
            ts = _ts_for_seed(seed)
        else:
            ts = _ts_for_seed(seed)
            aware_df, lagged_df, horizon_df = _run_backtest(
                ts, model, scaler, seed=seed, resume=resume, ckpt_every=args.ckpt_every,
                carbon_price_eur_per_t=args.carbon_price,
            )
            _save_seed_result(aware_df, lagged_df, horizon_df, seed)
            _clear_seed_partial(seed)

        # ----- Phase 2: CoolProp true-cost re-eval (per-strategy cached) --------
        # Mirrors scripts/07_carbon_sweep.py: each (seed, strategy) keeps a
        # consolidated "done" parquet plus an in-progress per-row partial.
        strat_dfs = {"aware": aware_df, "lagged": lagged_df, "horizon": horizon_df}
        true_costs: dict[str, pd.Series] = {}
        for s, df in strat_dfs.items():
            done_path = _seed_true_cost_done_path(seed, s)
            if done_path.exists() and resume:
                cached_tc = pd.read_parquet(done_path)
                if "time" in cached_tc.columns:
                    cached_tc["time"] = pd.to_datetime(cached_tc["time"], utc=True)
                    cached_tc = cached_tc.set_index("time")
                true_costs[s] = cached_tc["true_cost_eur"]
            else:
                true_costs[s] = _eval_true_cost_for_seed_strategy(
                    df, ts, seed, s, args.carbon_price, resume=resume,
                )
                done_path.parent.mkdir(parents=True, exist_ok=True)
                done_df = true_costs[s].reset_index()
                tmp = done_path.with_suffix(".parquet.tmp")
                done_df.to_parquet(tmp, index=False)
                tmp.replace(done_path)
                partial = _seed_true_cost_partial_path(seed, s)
                if partial.exists():
                    partial.unlink()

        # Yearly aggregation uses CoolProp truth, not PINN predictions.
        yearly_aware   = true_costs["aware"].resample("YE").sum()
        yearly_lagged  = true_costs["lagged"].resample("YE").sum()
        yearly_horizon = true_costs["horizon"].resample("YE").sum()

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
