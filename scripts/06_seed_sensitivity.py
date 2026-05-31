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

# Only torch-free modules are imported at module top. The parallel CoolProp
# workers re-import this module on spawn (Windows); pulling in torch here would
# make every one of them load the CUDA DLLs (cufft etc.), exhausting the
# paging file when many workers run at once. torch-dependent imports
# (baseline/dispatch/pinn) are therefore done lazily inside the functions that
# actually run dispatch — see _run_backtest / _process_one_seed / main.
from lng_pinn.composition import CARGO_CYCLE_DAYS, build_composition_series

# Defined locally (instead of imported from baseline) to keep this module's
# top-level import graph torch-free for the CoolProp workers.
COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
SEEDS = [42, 0, 1, 7, 13, 19, 23, 31, 37, 53]
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
    """Legacy per-row worker. Kept for back-compat; v1.5 uses
    ``_simulate_thermo_for_worker`` + driver-side cost arithmetic instead.
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


# ---- v1.5 E1: dedupe-before-submit thermo memoization -------------------------
def _simulate_thermo_for_worker(args: tuple) -> float | None:
    """Thermo-only worker — returns W_total (kWh/kg) for the bucketed args.
    Mirrors the helper in 07_carbon_sweep.py.
    """
    composition, m_dot, T_amb, T_sw = args
    from lng_pinn.plant import simulate

    try:
        return float(simulate(composition, m_dot, T_amb, T_sw).W_total)
    except ValueError:
        return None


def _thermo_key(
    comp: tuple,
    m_dot: float,
    T_amb: float,
    T_sw: float,
    m_dot_bucket: float = 0.5,
    T_bucket: float = 0.5,
) -> tuple:
    """Quantised key for thermo memoization. See lng-pinn-v1.5-plan.md §E1."""
    return (
        comp,
        round(m_dot / m_dot_bucket) * m_dot_bucket,
        round(T_amb / T_bucket) * T_bucket,
        round(T_sw / T_bucket) * T_bucket,
    )


# Canonical column order — must match scripts/07_carbon_sweep.py's
# _VALIDATION_COLS. Without an agreed schema the per-row CSV append from
# whichever script writes second silently mis-aligns values into the
# columns the other script chose.
_VALIDATION_COLS = [
    "script",
    "carbon_price_eur_per_t",
    "seed",
    "strategy",
    "n_total",
    "n_sampled",
    "mean_rel_err",
    "median_abs_rel_err",
    "p95_abs_rel_err",
    "max_abs_rel_err",
]


def _append_validation_diagnostics(
    *,
    seed: int,
    strategy: str,
    carbon_price: float,
    rel_err: np.ndarray,
    n_total: int,
    n_sampled: int,
) -> None:
    """Append a one-row PINN-vs-CoolProp validation summary."""
    if rel_err.size == 0:
        return
    path = RESULTS_DIR / "phase2_validation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "script": "06_seed_sensitivity",
        "carbon_price_eur_per_t": carbon_price,
        "seed": seed,
        "strategy": strategy,
        "n_total": int(n_total),
        "n_sampled": int(n_sampled),
        "mean_rel_err": float(np.mean(rel_err)),
        "median_abs_rel_err": float(np.median(np.abs(rel_err))),
        "p95_abs_rel_err": float(np.quantile(np.abs(rel_err), 0.95)),
        "max_abs_rel_err": float(np.max(np.abs(rel_err))),
    }])[_VALIDATION_COLS]
    header = not path.exists()
    row.to_csv(path, mode="a", index=False, header=header)


def _eval_true_cost_for_seed_strategy(
    dispatch_df: pd.DataFrame,
    ts: pd.DataFrame,
    seed: int,
    strategy: str,
    carbon_price: float,
    resume: bool = True,
    ckpt_every: int = 2000,
    n_workers: int | None = None,
    tqdm_position: int = 1,
    validation_sample_frac: float = 1.0,
) -> pd.Series:
    """CoolProp ground-truth cost (EUR/h) per hour for one (seed, strategy).

    v1.5 E1: rows are bucketed via ``_thermo_key`` so only unique buckets
    hit the CoolProp pool. W_total is broadcast back to all rows sharing a
    bucket; cost arithmetic happens in the driver.

    v1.5 E2: when ``validation_sample_frac < 1.0``, only that fraction
    of rows is evaluated through CoolProp; the rest use the PINN's predicted
    cost from ``dispatch_df.cost_eur``. Per-call validation rel-err
    diagnostics are appended to ``results/tables/phase2_validation.csv``.
    """
    from lng_pinn.thermo import co2_per_kg_fuel

    joined = dispatch_df.join(ts, how="inner")
    n = len(joined)
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 1)

    rows_factors: list[tuple] = []
    keys: list[tuple] = []
    for row in joined.itertuples():
        comp = tuple(float(getattr(row, c)) for c in COMP_COLS)
        m_dot = float(row.m_dot)
        T_amb = float(row.T_amb)
        T_sw = float(row.T_sw)
        price = float(row.price_eur_mwh)
        co2 = co2_per_kg_fuel(comp) if carbon_price > 0.0 else 0.0
        key = _thermo_key(comp, m_dot, T_amb, T_sw)
        keys.append(key)
        rows_factors.append((price, m_dot, co2))

    # E2: deterministic per (carbon_price, seed, strategy) sample.
    if validation_sample_frac >= 1.0:
        sample_mask = np.ones(n, dtype=bool)
    else:
        sd = (int(carbon_price * 1000) * 9973 + seed * 31 + hash(strategy) % 9973) & 0x7FFFFFFF
        rng = np.random.default_rng(sd)
        n_sample = max(1, int(round(n * validation_sample_frac)))
        sample_idx = np.sort(rng.choice(n, size=n_sample, replace=False))
        sample_mask = np.zeros(n, dtype=bool)
        sample_mask[sample_idx] = True

    partial_path = _seed_true_cost_partial_path(seed, strategy)
    done: dict[int, float] = {}
    if resume and partial_path.exists():
        try:
            prior = pd.read_parquet(partial_path)
            for r in prior.itertuples(index=False):
                done[int(r._row_idx)] = float(r.true_cost_eur)
        except Exception:
            done = {}

    pending_rows = [i for i in range(n) if i not in done and sample_mask[i]]

    if validation_sample_frac < 1.0:
        pinn_cost = joined["cost_eur"].to_numpy() if "cost_eur" in joined.columns else None
        if pinn_cost is None:
            raise RuntimeError(
                "validation_sample_frac < 1.0 requires dispatch_df.cost_eur "
                "(the PINN-predicted cost). It's missing from the joined frame."
            )
        for i in range(n):
            if not sample_mask[i] and i not in done:
                done[i] = float(pinn_cost[i])

    if pending_rows:
        key_to_rows: dict[tuple, list[int]] = {}
        for i in pending_rows:
            key_to_rows.setdefault(keys[i], []).append(i)
        unique_keys = list(key_to_rows.keys())
        worker_args = [(k[0], k[1], k[2], k[3]) for k in unique_keys]

        completed_since_ckpt = 0

        def _broadcast(key_idx: int, W: float | None) -> None:
            nonlocal completed_since_ckpt
            k = unique_keys[key_idx]
            for i in key_to_rows[k]:
                if W is None:
                    continue
                price, m_dot, co2 = rows_factors[i]
                elec = price * W * m_dot * 3.6
                carbon = carbon_price * co2 * m_dot * 3.6
                done[i] = float(elec + carbon)
                completed_since_ckpt += 1
                if completed_since_ckpt >= ckpt_every:
                    _flush_true_cost_partial(done, partial_path)
                    completed_since_ckpt = 0

        if n_workers <= 1:
            for idx, a in enumerate(tqdm(
                worker_args, total=len(worker_args),
                desc=f"  true-cost seed={seed} {strategy}", unit="key",
                position=tqdm_position, leave=False,
            )):
                _broadcast(idx, _simulate_thermo_for_worker(a))
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_simulate_thermo_for_worker, a): idx
                    for idx, a in enumerate(worker_args)
                }
                for fut in tqdm(
                    as_completed(futures), total=len(futures),
                    desc=f"  true-cost seed={seed} {strategy}", unit="key",
                    position=tqdm_position, leave=False,
                ):
                    _broadcast(futures[fut], fut.result())
        _flush_true_cost_partial(done, partial_path)

    if validation_sample_frac < 1.0:
        pinn_cost_arr = joined["cost_eur"].to_numpy()
        sample_indices = np.where(sample_mask)[0]
        sample_true = np.array(
            [done.get(int(i), np.nan) for i in sample_indices], dtype=np.float64
        )
        sample_pinn = pinn_cost_arr[sample_indices]
        denom = np.abs(sample_pinn) + 1e-12
        rel_err = (sample_true - sample_pinn) / denom
        rel_err = rel_err[np.isfinite(rel_err)]
        _append_validation_diagnostics(
            seed=seed, strategy=strategy, carbon_price=carbon_price,
            rel_err=rel_err, n_total=n, n_sampled=int(sample_mask.sum()),
        )

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
    tqdm_position: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Lazy torch-pulling imports — see the module-top comment.
    from lng_pinn.baseline import optimize_blind_horizon, optimize_blind_lagged
    from lng_pinn.dispatch import M_DOT_MAX, optimize

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

    pbar = tqdm(
        pending_starts, total=len(pending_starts),
        desc=f"  seed={seed} windows", position=tqdm_position, leave=False,
    )
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


def _process_one_seed(
    seed: int,
    no_resume: bool,
    ckpt_every: int,
    carbon_price: float,
    inner_workers: int,
    slot: int = 0,
    validation_sample_frac: float = 1.0,
) -> list[dict]:
    """Self-contained worker: dispatch + Phase-2 CoolProp re-eval for one seed.

    Loads the model and timeseries inside the worker so the parent doesn't
    pay for pickling them across the process boundary. ``slot`` gives this
    worker a unique tqdm row so concurrent workers don't overwrite each
    other's progress lines.

    Returns the seed's contribution to ``all_records`` (one dict per year).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    resume = not no_resume
    pos = slot + 1

    from lng_pinn.pinn import load  # lazy torch import — see module-top comment
    model, scaler = load()
    model.eval()

    cached = _load_seed_result(seed) if resume else None
    if cached is not None:
        aware_df, lagged_df, horizon_df = cached
        ts = _ts_for_seed(seed)
    else:
        ts = _ts_for_seed(seed)
        aware_df, lagged_df, horizon_df = _run_backtest(
            ts, model, scaler, seed=seed, resume=resume, ckpt_every=ckpt_every,
            carbon_price_eur_per_t=carbon_price, tqdm_position=pos,
        )
        _save_seed_result(aware_df, lagged_df, horizon_df, seed)
        _clear_seed_partial(seed)

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
                df, ts, seed, s, carbon_price, resume=resume,
                n_workers=inner_workers, tqdm_position=pos,
                validation_sample_frac=validation_sample_frac,
            )
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_df = true_costs[s].reset_index()
            tmp = done_path.with_suffix(".parquet.tmp")
            done_df.to_parquet(tmp, index=False)
            tmp.replace(done_path)
            partial = _seed_true_cost_partial_path(seed, s)
            if partial.exists():
                partial.unlink()

    yearly_aware   = true_costs["aware"].resample("YE").sum()
    yearly_lagged  = true_costs["lagged"].resample("YE").sum()
    yearly_horizon = true_costs["horizon"].resample("YE").sum()

    records = []
    for ts_end in yearly_aware.index:
        records.append({
            "seed": seed,
            "year": int(ts_end.year),
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
    return records


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
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Parallel processes across seeds. Default 1 (serial). Inner CoolProp "
            "pool auto-scales to floor(cpu_count / workers) per outer worker so "
            "total CPU usage stays sane."
        ),
    )
    parser.add_argument(
        "--validation-sample-frac", type=float, default=1.0,
        help=(
            "v1.5 E2: fraction of Phase-2 rows to evaluate through CoolProp "
            "(default 1.0 = full ground-truth re-eval). At 0.05 the remaining "
            "95%% of rows use the PINN's predicted cost (matches CoolProp to "
            "~1e-7 rel err per v1.4); validation rel-err stats are logged to "
            "results/tables/phase2_validation.csv."
        ),
    )
    args = parser.parse_args()
    resume = not args.no_resume

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    total_cores = max(1, os.cpu_count() or 1)
    n_workers = max(1, min(args.workers, len(SEEDS)))
    inner_workers = max(1, total_cores // n_workers)
    print(
        f"git_sha={git_sha}  seeds={SEEDS}  carbon_price={args.carbon_price:.1f} EUR/tCO2  "
        f"workers={n_workers} (inner CoolProp pool ≈ {inner_workers} per worker)"
    )

    all_records: list[dict] = []

    if n_workers > 1:
        # Parallel-seed path: each worker is self-contained (loads its own
        # model + timeseries). Outer tqdm tracks completed seeds; inner
        # per-worker progress lines live at positions slot+1.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _process_one_seed,
                    int(seed), args.no_resume, args.ckpt_every, args.carbon_price,
                    inner_workers, i % n_workers,
                    args.validation_sample_frac,
                ): int(seed)
                for i, seed in enumerate(SEEDS)
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Seeds (parallel)", unit="seed", position=0,
            ):
                seed = futures[future]
                try:
                    records = future.result()
                except Exception as exc:
                    tqdm.write(f"  seed={seed}: FAILED with {exc!r}")
                    raise
                all_records.extend(records)
                # Per-seed total saving printout (matches the serial path format).
                lagged_total = sum(r["blind_lagged_eur"] for r in records)
                horizon_total = sum(r["blind_horizon_eur"] for r in records)
                aware_total = sum(r["aware_eur"] for r in records)
                total_lagged_pct  = (lagged_total - aware_total)  / lagged_total  * 100
                total_horizon_pct = (horizon_total - aware_total) / horizon_total * 100
                tqdm.write(
                    f"  seed={seed}  lagged={total_lagged_pct:.2f}%  "
                    f"horizon={total_horizon_pct:.2f}%"
                )
    else:
        # Serial path — preserves the in-line per-seed prints and resume messages.
        from lng_pinn.pinn import load  # lazy torch import — see module-top comment
        model, scaler = load()
        model.eval()

        for seed in tqdm(SEEDS, desc="Seeds"):
            # ----- Phase 1: dispatch backtest -----------------------------------
            cached = _load_seed_result(seed) if resume else None
            if cached is not None:
                aware_df, lagged_df, horizon_df = cached
                print(f"  seed={seed}: reusing cached backtest result")
                ts = _ts_for_seed(seed)
            else:
                ts = _ts_for_seed(seed)
                aware_df, lagged_df, horizon_df = _run_backtest(
                    ts, model, scaler, seed=seed, resume=resume,
                    ckpt_every=args.ckpt_every,
                    carbon_price_eur_per_t=args.carbon_price,
                )
                _save_seed_result(aware_df, lagged_df, horizon_df, seed)
                _clear_seed_partial(seed)

            # ----- Phase 2: CoolProp true-cost re-eval (per-strategy cached) ----
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
                        validation_sample_frac=args.validation_sample_frac,
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
                        (yearly_lagged[ts_end] - yearly_aware[ts_end])
                        / yearly_lagged[ts_end] * 100
                    ),
                    "saving_vs_horizon_pct": float(
                        (yearly_horizon[ts_end] - yearly_aware[ts_end])
                        / yearly_horizon[ts_end] * 100
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
            print(
                f"  seed={seed}  lagged={total_lagged_pct:.2f}%  "
                f"horizon={total_horizon_pct:.2f}%"
            )

    results_df = pd.DataFrame(all_records)

    # Per-year mean ± std across seeds, for both baselines
    summary_rows = []
    for baseline in ["lagged", "horizon"]:
        col = f"saving_vs_{baseline}_pct"
        grp = results_df.groupby("year")[col].agg(mean="mean", std="std").reset_index()
        grp["baseline"] = baseline
        summary_rows.append(grp)
    summary = pd.concat(summary_rows, ignore_index=True)

    # v1.4 C0 — significance table. The paper's claim is about the *mean*
    # saving across composition seeds, so report SE = std/sqrt(n), the
    # one-sample t-stat against 0, and the two-sided p-value. Both per-year
    # and pooled (each seed's 3-year-mean saving, n = #seeds) rows are written.
    significance = _build_significance_table(results_df)
    significance.to_csv(RESULTS_DIR / "seed_significance.csv", index=False)

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
    print("Saved seed_sensitivity.csv, seed_sensitivity_summary.csv, seed_significance.csv")


def _build_significance_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per-year and pooled one-sample t-tests of aware-vs-baseline saving vs 0.

    The pooled row collapses each seed to its 3-year-mean saving first, so
    n = number of seeds and the rows are independent draws over composition
    realisations — the correct unit for the headline confidence interval.
    """
    from scipy import stats

    rows = []
    for baseline in ["lagged", "horizon"]:
        col = f"saving_vs_{baseline}_pct"

        for year, g in results_df.groupby("year"):
            rows.append(_ttest_row(g[col].to_numpy(), baseline, str(int(year)), stats))

        per_seed = results_df.groupby("seed")[col].mean().to_numpy()
        rows.append(_ttest_row(per_seed, baseline, "ALL_3yr_mean", stats))

    return pd.DataFrame(rows)


def _ttest_row(x: "np.ndarray", baseline: str, scope: str, stats: object) -> dict:
    """One-sample t-test summary dict for saving array ``x`` vs 0."""
    n = int(len(x))
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if (n > 1 and se > 0) else float("nan")  # type: ignore[attr-defined]
    return {
        "baseline": baseline,
        "scope": scope,
        "n": n,
        "mean_pct": round(mean, 4),
        "std_pct": round(std, 4),
        "se_pct": round(se, 4),
        "ci95_lo_pct": round(mean - 1.96 * se, 4),
        "ci95_hi_pct": round(mean + 1.96 * se, 4),
        "t_stat": round(t, 3),
        "p_two_sided": p,
    }


if __name__ == "__main__":
    main()
