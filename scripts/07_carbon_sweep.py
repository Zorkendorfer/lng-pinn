"""v1.3 carbon-price sweep — the headline figure.

Sweeps CO2 prices ∈ {0, 20, 40, 80, 120, 160} EUR/tCO2. For each price:
  1. Run dispatch + 5 baselines on the full 3-year timeseries via a
     programmatic entry point (no shelling out to 04_run_dispatch.py).
  2. Re-evaluate each strategy's schedule through CoolProp to get the
     true cost (same logic as 05_make_figures.py's build_true_cost_summary).
  3. Compute saving_vs_horizon_pct per year.

Results are cached per price to results/tables/carbon_sweep_co2_<price>.csv
so a partial sweep can resume cleanly. The final figure is written to
results/figures/fig6_carbon_sweep.pdf.

Cost: ~5–10 min per price point on M-series CPU once the model + dispatch
windows are warm. The sweep is embarrassingly parallel across prices but
this script runs them serially — the inner dispatch is already parallelised.
"""

from __future__ import annotations

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

# Only torch-free modules at module top: the parallel CoolProp workers re-import
# this module on spawn (Windows), and pulling in torch here would make every
# worker load the CUDA DLLs and exhaust the paging file. torch-dependent
# imports (baseline/dispatch/pinn) are done lazily in the dispatch functions.
# `plots` is torch-free (matplotlib/seaborn only) so it can stay at top.
from lng_pinn.composition import CARGO_CYCLE_DAYS
from lng_pinn.plots import fig_carbon_sweep

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]
DEFAULT_PRICES = (0.0, 20.0, 40.0, 80.0, 120.0, 160.0)
HORIZON_DAYS = 7
INV_INITIAL = 0.85
STRATEGIES = ("aware", "horizon", "lagged", "annual", "constant")


def _dispatch_partial_records_path(price: float) -> Path:
    """All-strategies-tagged record cache for an in-progress Phase-1 dispatch."""
    return PROCESSED_DIR / f"carbon_dispatch_partial_co2{int(price)}.parquet"


def _dispatch_partial_state_path(price: float) -> Path:
    """Inventories + next_start for the in-progress Phase-1 dispatch."""
    return PROCESSED_DIR / f"carbon_dispatch_partial_co2{int(price)}.json"


def _save_dispatch_partial(
    records_by_strategy: dict[str, list[dict]],
    inv: dict[str, float],
    next_start: int,
    price: float,
) -> None:
    """Atomic flush of Phase-1 dispatch state for one carbon price."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rec_path = _dispatch_partial_records_path(price)
    frames = []
    for strategy, records in records_by_strategy.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        df["_strategy"] = strategy
        frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        tmp = rec_path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(rec_path)
    state_path = _dispatch_partial_state_path(price)
    state = {"next_start": next_start, "inv": inv}
    tmp_state = state_path.with_suffix(".json.tmp")
    tmp_state.write_text(json.dumps(state))
    tmp_state.replace(state_path)


def _load_dispatch_partial(
    price: float,
) -> tuple[dict[str, list[dict]], dict[str, float], int] | None:
    state_path = _dispatch_partial_state_path(price)
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
        next_start = int(state["next_start"])
        inv = {k: float(v) for k, v in state["inv"].items()}
        records: dict[str, list[dict]] = {s: [] for s in STRATEGIES}
        rec_path = _dispatch_partial_records_path(price)
        if rec_path.exists():
            combined = pd.read_parquet(rec_path)
            for strategy, group in combined.groupby("_strategy", sort=False):
                records[str(strategy)] = group.drop(columns=["_strategy"]).to_dict(
                    orient="records"
                )
        return records, inv, next_start
    except Exception as exc:
        print(f"  co2={price:.0f}: could not load dispatch partial ({exc}); restarting")
        return None


def _clear_dispatch_partial(price: float) -> None:
    for path in (_dispatch_partial_state_path(price), _dispatch_partial_records_path(price)):
        if path.exists():
            path.unlink()


def _run_dispatch_for_price(
    ts: pd.DataFrame,
    model: object,
    scaler: object,
    carbon_price: float,
    tqdm_position: int = 1,
    resume: bool = True,
    ckpt_every: int = 20,
) -> dict[str, pd.DataFrame]:
    """Run all 5 strategies on the full timeseries at one carbon price.

    Returns dict[strategy -> DataFrame with columns m_dot, cost_eur,
    indexed by time]. The returned cost_eur is the PINN's prediction
    (electricity + carbon term); true-cost evaluation happens in a
    second pass via CoolProp.

    Per-window partial is flushed every ``ckpt_every`` windows to
    ``carbon_dispatch_partial_co2<price>.{parquet,json}``. On resume the
    partial is loaded, inventories are restored, and only the missing
    windows are processed. Cargo deliveries fire deterministically from
    ``start % cargo_cycle_hours == 0``, so resume is safe regardless of
    where the previous run stopped.

    ``tqdm_position`` controls the terminal row the per-window bar occupies —
    siblings in parallel-price mode pass distinct positions so their bars
    stack instead of overwriting each other.
    """
    # Lazy torch-pulling imports — see the module-top comment.
    from lng_pinn.baseline import (
        optimize_blind_annual,
        optimize_blind_horizon,
        optimize_blind_lagged,
        optimize_constant_flow,
    )
    from lng_pinn.dispatch import M_DOT_MAX, optimize

    H = HORIZON_DAYS * 24
    step = 24
    cargo_cycle_hours = CARGO_CYCLE_DAYS * 24
    cargo_amount = 0.55  # fraction of TANK_CAP per cargo — matches 04_run_dispatch.py

    starts = list(range(0, len(ts) - H + 1, step))
    demand_kg = M_DOT_MAX * 0.6 * H * 3600
    annual_composition = ts[COMP_COLS].mean()

    records: dict[str, list[dict]] = {s: [] for s in STRATEGIES}
    inv = {s: INV_INITIAL for s in STRATEGIES}
    resume_start = 0

    if resume:
        loaded = _load_dispatch_partial(carbon_price)
        if loaded is not None:
            records, inv, resume_start = loaded
            n_recs = next(iter(records.values()), [])
            tqdm.write(
                f"  co2={carbon_price:.0f}: resuming Phase 1 with {len(n_recs)} hours "
                f"per strategy cached; next start={resume_start}"
            )

    pending_starts = [s for s in starts if s >= resume_start]
    flush_counter = 0
    last_completed = resume_start

    pbar = tqdm(
        pending_starts, desc=f"  dispatch co2={carbon_price:.0f}", unit="day",
        position=tqdm_position, leave=False,
    )
    for start in pbar:
        if start > 0 and start % cargo_cycle_hours == 0:
            for s in STRATEGIES:
                inv[s] = min(0.92, inv[s] + cargo_amount)

        window = ts.iloc[start : start + H]
        lagged_composition = ts[COMP_COLS].iloc[start]
        n_record = min(step, len(window))

        cp = carbon_price
        scheds = {
            "aware": optimize(
                window, model, scaler, demand_kg, inv["aware"],
                carbon_price_eur_per_t=cp,
            ),
            "horizon": optimize_blind_horizon(
                window, model, scaler, demand_kg, inv["horizon"],
                carbon_price_eur_per_t=cp,
            ),
            "lagged": optimize_blind_lagged(
                window, model, scaler, demand_kg, lagged_composition, inv["lagged"],
                carbon_price_eur_per_t=cp,
            ),
            "annual": optimize_blind_annual(
                window, model, scaler, demand_kg, annual_composition, inv["annual"],
                carbon_price_eur_per_t=cp,
            ),
            "constant": optimize_constant_flow(
                window, model, scaler, demand_kg, inv["constant"],
                carbon_price_eur_per_t=cp,
            ),
        }

        for s, sched in scheds.items():
            for t, ts_row in enumerate(window.iloc[:n_record].itertuples()):
                records[s].append({
                    "time": ts_row.Index,
                    "m_dot": float(sched.m_dot[t]),
                    "cost_eur": float(sched.cost_eur[t]),
                })
            inv[s] = float(sched.tank_level[n_record])

        last_completed = start + step
        flush_counter += 1
        if flush_counter >= ckpt_every:
            _save_dispatch_partial(records, inv, last_completed, carbon_price)
            flush_counter = 0

    if pending_starts:
        _save_dispatch_partial(records, inv, last_completed, carbon_price)

    out: dict[str, pd.DataFrame] = {}
    for s in STRATEGIES:
        df = pd.DataFrame(records[s])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
        out[s] = df
    return out


def _per_strategy_done_path(price: float, strategy: str) -> Path:
    """Completed per-strategy true-cost cache for one (price, strategy)."""
    return PROCESSED_DIR / f"true_costs_co2{int(price)}_{strategy}.parquet"


def _per_strategy_partial_path(price: float, strategy: str) -> Path:
    """In-progress per-row true-cost cache, flushed every K completions."""
    return PROCESSED_DIR / f"true_costs_co2{int(price)}_{strategy}_inprogress.parquet"


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
    """Worker: one CoolProp simulation + cost calculation, returns EUR/h.

    Top-level function so ProcessPoolExecutor can pickle it. Locally imports
    plant/thermo so worker startup doesn't pay for the parent's full import
    graph.
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


def _true_cost_for_strategy(
    dispatch_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    carbon_price: float,
    label: str = "",
    n_workers: int | None = None,
    tqdm_position: int = 1,
    resume: bool = True,
    ckpt_every: int = 2000,
) -> pd.Series:
    """CoolProp ground-truth cost (electricity + carbon) per hour.

    Parallelised across processes. Falls back to serial when ``n_workers == 1``
    (useful for debugging and tiny inputs). ``tqdm_position`` matches the
    dispatch bar's position so siblings in parallel-price mode each keep a
    single, stable terminal row.

    Resume behaviour: every ``ckpt_every`` completed rows are flushed to
    ``true_costs_co2<price>_<label>_inprogress.parquet`` keyed by positional
    row index. On rerun the partial is loaded and only missing rows are
    submitted to the pool. The partial is removed once the caller persists
    the consolidated per-strategy result.
    """
    joined = dispatch_df.join(ts_df, how="inner")
    n = len(joined)
    desc = f"  true-cost {label}" if label else "  true-cost"
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))

    # Build all per-row args up front (cheap; ~26k rows × small tuples).
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

    partial_path = _per_strategy_partial_path(carbon_price, label) if label else None
    done: dict[int, float] = {}
    if resume and partial_path is not None and partial_path.exists():
        try:
            prior = pd.read_parquet(partial_path)
            for r in prior.itertuples(index=False):
                done[int(r._row_idx)] = float(r.true_cost_eur)
        except Exception:
            done = {}

    pending = [(i, a) for i, a in enumerate(arg_list) if i not in done]
    if pending:
        completed_since_ckpt = 0
        if n_workers <= 1:
            for i, a in tqdm(
                pending, total=len(pending), desc=desc, unit="hr",
                position=tqdm_position, leave=False,
            ):
                cost = _true_cost_row(a)
                if cost is not None:
                    done[i] = cost
                completed_since_ckpt += 1
                if completed_since_ckpt >= ckpt_every and partial_path is not None:
                    _flush_true_cost_partial(done, partial_path)
                    completed_since_ckpt = 0
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_true_cost_row, a): i for i, a in pending}
                for future in tqdm(
                    as_completed(futures), total=len(pending),
                    desc=desc, unit="hr", position=tqdm_position, leave=False,
                ):
                    i = futures[future]
                    cost = future.result()
                    if cost is not None:
                        done[i] = cost
                    completed_since_ckpt += 1
                    if completed_since_ckpt >= ckpt_every and partial_path is not None:
                        _flush_true_cost_partial(done, partial_path)
                        completed_since_ckpt = 0
        if partial_path is not None:
            _flush_true_cost_partial(done, partial_path)

    out = [done.get(i, np.nan) for i in range(n)]
    return pd.Series(out, index=joined.index, name="true_cost_eur")


def _process_one_price(
    price: float,
    no_resume: bool,
    inner_workers: int,
    slot: int = 0,
    ckpt_every: int = 20,
) -> pd.DataFrame:
    """Self-contained worker: dispatch + true-cost re-eval for one carbon price.

    Loads the model and timeseries inside the worker so the parent doesn't
    have to pickle them across the process boundary. ``slot`` is the worker's
    assigned tqdm row (position = slot + 1; the parent's "Prices" bar owns
    position 0), letting every concurrent worker keep its own stable progress
    line instead of overwriting siblings. ``ckpt_every`` controls how often
    Phase-1 dispatch flushes its per-window partial.
    """
    cache_path = RESULTS_DIR / f"carbon_sweep_co2_{int(price)}.csv"
    if cache_path.exists() and not no_resume:
        yearly = pd.read_csv(cache_path)
        if "price_co2_eur_per_t" not in yearly.columns:
            yearly["price_co2_eur_per_t"] = price
        return yearly

    from lng_pinn.pinn import load  # lazy torch import — see module-top comment
    model, scaler = load()
    model.eval()
    ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts.index = pd.to_datetime(ts.index, utc=True)

    pos = slot + 1
    scheds = _run_dispatch_for_price(
        ts, model, scaler, price, tqdm_position=pos,
        resume=not no_resume, ckpt_every=ckpt_every,
    )

    true_costs: dict[str, pd.Series] = {}
    for s in STRATEGIES:
        done_path = _per_strategy_done_path(price, s)
        if done_path.exists() and not no_resume:
            cached = pd.read_parquet(done_path)
            if "time" in cached.columns:
                cached["time"] = pd.to_datetime(cached["time"], utc=True)
                cached = cached.set_index("time")
            true_costs[s] = cached["true_cost_eur"]
        else:
            true_costs[s] = _true_cost_for_strategy(
                scheds[s], ts, price, label=s,
                n_workers=inner_workers, tqdm_position=pos,
                resume=not no_resume,
            )
            # Persist the consolidated per-strategy result so a Ctrl-C between
            # strategies doesn't lose this one's CoolProp work.
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_df = true_costs[s].reset_index()
            tmp = done_path.with_suffix(".parquet.tmp")
            done_df.to_parquet(tmp, index=False)
            tmp.replace(done_path)
            # The in-progress per-row partial is now redundant.
            partial = _per_strategy_partial_path(price, s)
            if partial.exists():
                partial.unlink()

    yearly = _yearly_savings(true_costs)
    yearly["price_co2_eur_per_t"] = price
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(cache_path, index=False)

    # Final yearly CSV is the consolidated cache; per-strategy parquets and
    # the Phase-1 dispatch partial are now redundant for this price.
    for s in STRATEGIES:
        for path in (_per_strategy_done_path(price, s), _per_strategy_partial_path(price, s)):
            if path.exists():
                path.unlink()
    _clear_dispatch_partial(price)

    return yearly


def _yearly_savings(true_costs: dict[str, pd.Series]) -> pd.DataFrame:
    """Aware-vs-horizon true-cost saving per year."""
    yearly = pd.DataFrame({s: true_costs[s].resample("YE").sum() for s in STRATEGIES}).dropna()
    yearly["saving_vs_horizon_pct"] = (
        (yearly["horizon"] - yearly["aware"]) / yearly["horizon"] * 100
    )
    yearly["saving_vs_lagged_pct"] = (
        (yearly["lagged"] - yearly["aware"]) / yearly["lagged"] * 100
    )
    yearly = yearly.reset_index()
    yearly["year"] = pd.to_datetime(yearly["time"], utc=True).dt.year
    return yearly.drop(columns=["time"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prices", type=float, nargs="+", default=list(DEFAULT_PRICES),
        help="CO2 prices in EUR/tCO2 to sweep over.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore per-price cached CSVs and recompute every point.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Parallel processes across carbon prices. Default 1 (serial). "
            "Inner CoolProp pool auto-scales to floor(cpu_count / workers) per "
            "outer worker so total CPU usage stays sane."
        ),
    )
    parser.add_argument(
        "--ckpt-every", type=int, default=20,
        help="Flush Phase-1 dispatch partial state every K windows (default 20).",
    )
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    total_cores = max(1, os.cpu_count() or 1)
    n_workers = max(1, min(args.workers, len(args.prices)))
    inner_workers = max(1, total_cores // n_workers)
    print(
        f"git_sha={git_sha}  prices={args.prices}  "
        f"workers={n_workers} (inner CoolProp pool ≈ {inner_workers} per worker)"
    )

    all_rows: list[pd.DataFrame] = []

    if n_workers == 1:
        # Serial path — keep the rich per-phase logging.
        from lng_pinn.pinn import load  # lazy torch import — see module-top comment
        model, scaler = load()
        model.eval()
        ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
        ts.index = pd.to_datetime(ts.index, utc=True)

        price_pbar = tqdm(args.prices, desc="Prices", unit="price", position=0)
        for price in price_pbar:
            price_pbar.set_postfix_str(f"co2={price:.0f} EUR/t")
            cache_path = RESULTS_DIR / f"carbon_sweep_co2_{int(price)}.csv"
            if cache_path.exists() and not args.no_resume:
                tqdm.write(f"  co2={price:.0f}: using cached {cache_path.name}")
                yearly = pd.read_csv(cache_path)
            else:
                tqdm.write(f"  co2={price:.0f}: phase 1/2 — running dispatch backtest...")
                scheds = _run_dispatch_for_price(
                    ts, model, scaler, price,
                    resume=not args.no_resume, ckpt_every=args.ckpt_every,
                )
                tqdm.write(
                    f"  co2={price:.0f}: phase 2/2 — CoolProp re-evaluation of 5 strategies..."
                )
                true_costs: dict[str, pd.Series] = {}
                for s in STRATEGIES:
                    done_path = _per_strategy_done_path(price, s)
                    if done_path.exists() and not args.no_resume:
                        cached = pd.read_parquet(done_path)
                        if "time" in cached.columns:
                            cached["time"] = pd.to_datetime(cached["time"], utc=True)
                            cached = cached.set_index("time")
                        true_costs[s] = cached["true_cost_eur"]
                        tqdm.write(f"    {s}: loaded cached per-strategy result")
                    else:
                        true_costs[s] = _true_cost_for_strategy(
                            scheds[s], ts, price, label=s,
                            resume=not args.no_resume,
                        )
                        done_path.parent.mkdir(parents=True, exist_ok=True)
                        done_df = true_costs[s].reset_index()
                        tmp = done_path.with_suffix(".parquet.tmp")
                        done_df.to_parquet(tmp, index=False)
                        tmp.replace(done_path)
                        partial = _per_strategy_partial_path(price, s)
                        if partial.exists():
                            partial.unlink()
                yearly = _yearly_savings(true_costs)
                yearly["price_co2_eur_per_t"] = price
                tqdm.write(f"  co2={price:.0f}: done. Saving {cache_path.name}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                yearly.to_csv(cache_path, index=False)
                for s in STRATEGIES:
                    for path in (
                        _per_strategy_done_path(price, s),
                        _per_strategy_partial_path(price, s),
                    ):
                        if path.exists():
                            path.unlink()
                _clear_dispatch_partial(price)
            if "price_co2_eur_per_t" not in yearly.columns:
                yearly["price_co2_eur_per_t"] = price
            all_rows.append(yearly)
    else:
        # Parallel-price path — each worker is self-contained (loads its own
        # model + timeseries). Inner tqdm bars are silenced; only the outer
        # "Prices" bar advances as workers complete.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Slot = price index modulo n_workers — gives each concurrently
            # running worker a distinct tqdm row even when more prices than
            # workers exist (later tasks reuse a freed row).
            futures = {
                executor.submit(
                    _process_one_price,
                    float(price), args.no_resume, inner_workers,
                    i % n_workers, args.ckpt_every,
                ): float(price)
                for i, price in enumerate(args.prices)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Prices (parallel)",
                unit="price",
                position=0,
            ):
                price = futures[future]
                try:
                    yearly = future.result()
                except Exception as exc:
                    tqdm.write(f"  co2={price:.0f}: FAILED with {exc!r}")
                    raise
                tqdm.write(f"  co2={price:.0f}: done")
                all_rows.append(yearly)

    sweep_df = pd.concat(all_rows, ignore_index=True)
    # Sort so the figure / CSV are in price order regardless of completion order.
    if "price_co2_eur_per_t" in sweep_df.columns:
        sweep_df = sweep_df.sort_values(["price_co2_eur_per_t", "year"]).reset_index(drop=True)
    sweep_df.to_csv(RESULTS_DIR / "carbon_sweep.csv", index=False)
    fig_carbon_sweep(sweep_df)
    print("Saved results/figures/fig6_carbon_sweep.pdf and results/tables/carbon_sweep.csv")


if __name__ == "__main__":
    main()
