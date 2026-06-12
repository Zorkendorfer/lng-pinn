#!/usr/bin/env python3
"""
09_mixing_sensitivity.py
Robustness of the aware-vs-lagged saving to the in-tank mixing model.
(paper: section "Robustness to the tank-mixing model")

The headline aware-vs-lagged saving is generated *entirely* by how the in-tank
composition transitions after a cargo arrival. The paper default is a linear
blend over tau_mix = 5 d. This script reruns the existing 10-seed backtest for a
grid of (tau_mix, kernel) and writes results/tables/mixing_sensitivity.csv with
the 5-year-mean saving and 95% t-CI per cell. The aggregation is seed-first:
each seed is collapsed to its five-year mean before the t-test, so n is the
number of cargo-schedule seeds, not the number of seed-year cells.

WIRING (now done). The two `ADAPT:` hooks are wired to the project's own cargo
schedule + dispatch:

  * load_arrivals(seed, year) reconstructs the *exact* cargo schedule that
    lng_pinn.composition.build_composition_series draws for `seed` (same RNG
    draw order, same archetypes), then returns the raw step arrivals that fall
    in `year` (pre-year arrivals are kept at negative local hours so the
    year-boundary transition is reconstructed faithfully, not truncated). With
    kernel="linear", tau=5 d this reproduces build_composition_series sliced to
    the year to ~1e-16 -- verify with `--self-check`.

  * run_backtest_saving(comp_traj, seed, year, carbon_price) runs the same
    rolling-horizon aware-vs-lagged backtest as scripts/06, but per year and
    with the injected composition trajectory. It reports the saving from the
    PINN dispatch cost (the objective the optimiser actually minimises); the
    v1.4 validation showed the PINN reproduces the CoolProp ground-truth cost to
    ~1e-7 rel. err., so the expensive CoolProp re-eval that 06 runs is omitted
    here and the saving is unchanged to ~1e-6.

Because the backtest is run per year with a fresh tank (06 runs one continuous
backtest and resamples by year), the (tau=5, linear) cell reproduces the
headline up to small inventory-reset edge effects, not bit-identically.

    uv run python scripts/09_mixing_sensitivity.py --self-check
    uv run python scripts/09_mixing_sensitivity.py --carbon-price 80 --workers 10
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Manager
from pathlib import Path

# Keep each worker single-threaded: with --workers 10 on a 28-core box the
# parallelism must come from processes, not from torch/BLAS threads, or the LP
# solves thrash. Also force CPU -- dispatch is a tiny PINN forward + a HiGHS LP,
# so a GPU buys nothing and 10 CUDA contexts would exhaust the Windows paging
# file (the WinError 1455 we hit with the CoolProp workers). Set before any
# (lazy) torch import; spawned workers re-import this module and re-apply these.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Only torch-free modules at module top -- the parallel workers re-import this
# module on spawn (Windows), and pulling torch in here would load it in every
# one of them. torch-dependent imports (pinn/dispatch/baseline) are lazy, inside
# the functions that actually run dispatch. See scripts/06 for the same pattern.
from lng_pinn.composition import ARCHETYPES, BLEND_DAYS, CARGO_CYCLE_DAYS

DEFAULT_SEEDS = [42, 0, 1, 7, 13, 19, 23, 31, 37, 53]
DEFAULT_YEARS = [2021, 2022, 2023, 2024, 2025]

# Backtest constants -- must match scripts/06_seed_sensitivity.py so the
# (tau=5, linear) cell reproduces the headline.
COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]
HORIZON_DAYS = 7
DEMAND_FACTOR = 0.6
CARGO_CYCLE_HOURS = int(CARGO_CYCLE_DAYS * 24)
CARGO_AMOUNT = 0.55
INV0 = 0.85
INV_CAP = 0.92

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
CELLS_CACHE = PROCESSED_DIR / "mixing_sensitivity_cells.parquet"


# --------------------------------------------------------------------------- #
# Mixing kernels  (self-contained, simplex-preserving)                        #
# --------------------------------------------------------------------------- #
def blend_weight(dt_days: np.ndarray, tau_days: float, kernel: str) -> np.ndarray:
    """Weight w in [0,1] on the INCOMING cargo, dt_days after its arrival.
    composition(dt) = (1 - w) * x_anchor + w * x_in."""
    dt = np.asarray(dt_days, dtype=float)
    if kernel == "linear":          # ramp to full incoming over tau, then hold
        return np.clip(dt / tau_days, 0.0, 1.0)
    if kernel == "exp":             # first-order / CSTR; tau = e-folding time
        return 1.0 - np.exp(-dt / tau_days)
    if kernel == "step":            # limiting case: instantaneous switch
        return (dt >= 0.0).astype(float)
    raise ValueError(f"unknown kernel {kernel!r}")


def build_blended_trajectory(arrivals, n_hours: int, tau_days: float,
                             kernel: str) -> np.ndarray:
    """arrivals: list of (arrival_hour:int, x_in:np.ndarray[6]) sorted by hour.
    Returns comp[n_hours, 6] on the simplex. The anchor for each segment is the
    in-tank composition at the moment of that arrival, so chained, not-yet-
    settled transitions compose correctly. Each segment is a convex combination
    of (anchor, incoming), so mole fractions sum to 1 and stay >= 0 throughout.
    """
    comp = np.zeros((n_hours, 6), dtype=float)
    x_anchor = np.asarray(arrivals[0][1], dtype=float).copy()
    for j, (a_hour, x_in) in enumerate(arrivals):
        x_in = np.asarray(x_in, dtype=float)
        end = arrivals[j + 1][0] if j + 1 < len(arrivals) else n_hours
        seg = np.arange(max(a_hour, 0), max(end, 0))
        if seg.size:
            w = blend_weight((seg - a_hour) / 24.0, tau_days, kernel)
            comp[seg] = (1 - w)[:, None] * x_anchor[None, :] + w[:, None] * x_in[None, :]
        # anchor for the NEXT segment = composition at this segment's end hour
        w_end = blend_weight((end - a_hour) / 24.0, tau_days, kernel)
        x_anchor = (1 - w_end) * x_anchor + w_end * x_in
    if arrivals[0][0] > 0:          # hours before the first arrival
        comp[: arrivals[0][0]] = np.asarray(arrivals[0][1], dtype=float)
    return comp


# --------------------------------------------------------------------------- #
# Lazy singletons (per process): trained PINN + the full timeseries.          #
# --------------------------------------------------------------------------- #
_MODEL = None
_SCALER = None
_TS: pd.DataFrame | None = None

# Window-level progress. Each backtest ticks once per committed window so the
# bar moves every ~0.5 s instead of only when a whole year-backtest finishes.
# Parallel workers push to a shared Manager queue drained by the main process;
# the serial path updates a local tqdm directly.
_PROGRESS_Q = None
_SERIAL_BAR = None


def _tick() -> None:
    if _PROGRESS_Q is not None:
        try:
            _PROGRESS_Q.put_nowait(1)
        except Exception:
            pass
    elif _SERIAL_BAR is not None:
        _SERIAL_BAR.update(1)


def _init_worker(progress_queue) -> None:
    """ProcessPoolExecutor initializer: give each worker the shared progress
    queue and the src path (spawn re-imports this module fresh)."""
    global _PROGRESS_Q
    _PROGRESS_Q = progress_queue
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _get_model():
    """Load the trained PINN once per process (CPU; see module-top env)."""
    global _MODEL, _SCALER
    if _MODEL is None:
        import torch

        from lng_pinn.pinn import load
        torch.set_num_threads(1)
        _MODEL, _SCALER = load()
        _MODEL.eval()
    return _MODEL, _SCALER


def _get_timeseries() -> pd.DataFrame:
    """Full price/weather timeseries, UTC-indexed. Loaded once per process.

    This is the same file scripts/06 reads; composition columns present in it
    are ignored here -- the sweep injects its own trajectory."""
    global _TS
    if _TS is None:
        ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
        ts.index = pd.to_datetime(ts.index, utc=True)
        _TS = ts
    return _TS


# --------------------------------------------------------------------------- #
# Cargo schedule reconstruction (mirrors composition.build_composition_series) #
# --------------------------------------------------------------------------- #
def _global_cargo_schedule(seed: int, n_total: int):
    """Reconstruct the raw cargo arrivals build_composition_series would draw.

    Returns list[(global_hour:int, x:np.ndarray[6])]. The first element is the
    initial in-tank cargo (the anchor at hour 0); each subsequent element is a
    cargo arriving at a multiple of CARGO_CYCLE_HOURS. The RNG draw order is
    identical to build_composition_series (one draw for the initial fill, then
    one per cargo boundary while hour < n_total), so the archetype sequence --
    and hence the per-year composition -- matches exactly.
    """
    rng = np.random.default_rng(seed)
    comps = [np.asarray(ARCHETYPES[k], dtype=float) for k in ARCHETYPES]
    n_arch = len(comps)
    d0 = int(rng.integers(n_arch))          # initial current_idx
    arrivals = [(0, comps[d0])]             # in-tank fill (anchor)
    h = 0
    while h < n_total:
        d = int(rng.integers(n_arch))       # next_idx drawn at each cargo boundary
        arrivals.append((h, comps[d]))
        h += CARGO_CYCLE_HOURS
    return arrivals


# --------------------------------------------------------------------------- #
# ADAPT 1 (wired): per-(seed, year) raw step arrivals + horizon length.        #
# --------------------------------------------------------------------------- #
def load_arrivals(seed: int, year: int):
    """Return (arrivals, n_hours) for one (seed, year).

    arrivals: list of (arrival_hour:int, x_in:np.ndarray[6]) sorted by hour,
              hours indexed from the start of `year`'s backtest horizon. Cargoes
              that arrived *before* the year are kept at negative local hours so
              build_blended_trajectory reconstructs a transition already in
              progress at the year boundary (rather than restarting it). No
              blending is applied here -- these are raw step arrivals; tau/kernel
              enter only through build_blended_trajectory.
    n_hours:  number of hours of `year` in the timeseries.
    """
    ts = _get_timeseries()
    idx = ts.index
    n_total = len(idx)
    year_pos = np.where(idx.year == year)[0]
    if year_pos.size == 0:
        raise ValueError(f"year {year} not present in timeseries")
    start_pos = int(year_pos[0])
    n_hours = int(year_pos.size)

    global_arrivals = _global_cargo_schedule(seed, n_total)
    arrivals = [
        (int(g_hour - start_pos), vec)
        for g_hour, vec in global_arrivals
        if (g_hour - start_pos) < n_hours
    ]
    return arrivals, n_hours


# --------------------------------------------------------------------------- #
# ADAPT 2 (wired): rolling-horizon aware-vs-lagged backtest, comp injected.    #
# --------------------------------------------------------------------------- #
def run_backtest_saving(comp_traj: np.ndarray, seed: int, year: int,
                        carbon_price: float) -> float:
    """Aware-vs-lagged saving (%) for one (seed, year) under composition
    `comp_traj` (shape [n_hours, 6]).

    Mirrors scripts/06._run_backtest restricted to one year: 7-day window,
    24-h commit step, demand = M_DOT_MAX * 0.6 per hour, tank starts at 0.85 and
    is topped up by 0.55 (capped 0.92) every cargo cycle. 'aware' reads the
    evolving composition from the window; 'lagged' freezes it at the window
    start. saving% = 100 * (cost_lagged - cost_aware) / cost_lagged, on the PINN
    dispatch cost (the optimiser's own objective; ~1e-7 from CoolProp truth).
    """
    from lng_pinn.baseline import optimize_blind_lagged
    from lng_pinn.dispatch import M_DOT_MAX, optimize

    model, scaler = _get_model()
    ts_full = _get_timeseries()
    ts_year = ts_full.loc[ts_full.index.year == year].copy()
    n_hours = len(ts_year)
    if comp_traj.shape[0] != n_hours:
        raise ValueError(
            f"comp_traj has {comp_traj.shape[0]} hours but year {year} has {n_hours}"
        )
    for j, col in enumerate(COMP_COLS):
        ts_year[col] = comp_traj[:, j]

    H = HORIZON_DAYS * 24
    step = 24
    starts = list(range(0, n_hours - H + 1, step))
    demand_kg = M_DOT_MAX * DEMAND_FACTOR * H * 3600.0

    inv = {"aware": INV0, "lagged": INV0}
    cost_aware = 0.0
    cost_lagged = 0.0
    cp = carbon_price
    for start in starts:
        if start > 0 and start % CARGO_CYCLE_HOURS == 0:
            for s in inv:
                inv[s] = min(INV_CAP, inv[s] + CARGO_AMOUNT)

        window = ts_year.iloc[start : start + H]
        lagged_composition = ts_year[COMP_COLS].iloc[start]
        n = min(step, len(window))

        a_sched = optimize(  # type: ignore[arg-type]
            window, model, scaler, demand_kg, inv["aware"], carbon_price_eur_per_t=cp,
        )
        l_sched = optimize_blind_lagged(  # type: ignore[arg-type]
            window, model, scaler, demand_kg, lagged_composition, inv["lagged"],
            carbon_price_eur_per_t=cp,
        )

        cost_aware += float(a_sched.cost_eur[:n].sum())
        cost_lagged += float(l_sched.cost_eur[:n].sum())
        inv["aware"] = float(a_sched.tank_level[n])
        inv["lagged"] = float(l_sched.tank_level[n])
        _tick()

    if cost_lagged <= 0.0:
        return float("nan")
    return 100.0 * (cost_lagged - cost_aware) / cost_lagged


# --------------------------------------------------------------------------- #
# Sweep + aggregation                                                          #
# --------------------------------------------------------------------------- #
def _aggregate(tau_days, kernel, records) -> dict:
    """Seed-first aggregation: mean, SE, t/Wilcoxon vs 0, 95% t-CI."""
    df = pd.DataFrame(records)
    if df.empty:
        return {}
    df = df[np.isfinite(df["saving_pct"].astype(float))]
    seed_means = df.groupby("seed")["saving_pct"].mean().to_numpy(dtype=float)
    seed_means = seed_means[np.isfinite(seed_means)]
    n = int(seed_means.size)
    n_cells = int(len(df))
    mean = float(seed_means.mean()) if n else float("nan")
    std = float(seed_means.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if (n > 1 and se > 0) else float("nan")
    if n > 1 and se > 0:
        lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    else:
        lo, hi = mean, mean
    try:
        wilcoxon_p = (
            float(stats.wilcoxon(seed_means, zero_method="wilcox").pvalue)
            if n > 0 and not np.allclose(seed_means, 0.0)
            else float("nan")
        )
    except ValueError:
        wilcoxon_p = float("nan")
    return dict(
        tau_days=tau_days,
        kernel=kernel,
        n=n,
        n_seed_year_cells=n_cells,
        mean_pct=mean,
        std_pct=std,
        se_pct=se,
        ci_low=lo,
        ci_high=hi,
        t=t,
        p=p,
        wilcoxon_p_two_sided=wilcoxon_p,
    )


def cell(tau_days, kernel, seeds, years, carbon_price) -> dict:
    """One (tau, kernel) cell: collect seed-year savings, return seed-first stats.

    Kept as the reference serial implementation; main() uses a cached,
    optionally parallel task loop that feeds the identical vals into
    `_aggregate`."""
    records = []
    for seed in seeds:
        for year in years:
            arrivals, n_hours = load_arrivals(seed, year)
            traj = build_blended_trajectory(arrivals, n_hours, tau_days, kernel)
            records.append(
                {
                    "seed": seed,
                    "year": year,
                    "saving_pct": run_backtest_saving(traj, seed, year, carbon_price),
                }
            )
    return _aggregate(tau_days, kernel, records)


# --------------------------------------------------------------------------- #
# Per-(tau, kernel, seed, year) task + resumable cache                         #
# --------------------------------------------------------------------------- #
def _saving_task(args):
    """Picklable worker: build the trajectory for one cell-coordinate and
    return its saving. Returns (tau, kernel, seed, year, saving_pct)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    tau, kernel, seed, year, carbon_price = args
    try:
        arrivals, n_hours = load_arrivals(seed, year)
        traj = build_blended_trajectory(arrivals, n_hours, tau, kernel)
        val = float(run_backtest_saving(traj, seed, year, carbon_price))
    except Exception as exc:
        raise RuntimeError(
            f"mixing task failed: tau={tau:g}, kernel={kernel}, "
            f"seed={seed}, year={year}, carbon_price={carbon_price:g}"
        ) from exc
    return (tau, kernel, seed, year, val)


def _cell_key(tau, kernel, seed, year):
    return (round(float(tau), 6), str(kernel), int(seed), int(year))


def _safe_replace(src: Path, dst: Path, attempts: int = 20, delay: float = 0.2) -> None:
    """Atomic rename with retry -- works around transient Windows file locks."""
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def _load_cells_cache() -> dict:
    if not CELLS_CACHE.exists():
        return {}
    try:
        df = pd.read_parquet(CELLS_CACHE)
    except Exception:
        return {}
    out = {}
    for r in df.itertuples(index=False):
        out[_cell_key(r.tau_days, r.kernel, r.seed, r.year)] = float(r.saving_pct)
    return out


def _flush_cells_cache(done: dict) -> None:
    if not done:
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {"tau_days": k[0], "kernel": k[1], "seed": k[2], "year": k[3], "saving_pct": v}
        for k, v in done.items()
    ]
    df = pd.DataFrame(rows).sort_values(["kernel", "tau_days", "seed", "year"])
    tmp = CELLS_CACHE.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    _safe_replace(tmp, CELLS_CACHE)


def _self_check(seed: int = 42) -> None:
    """Confirm load_arrivals + build_blended_trajectory reproduce
    build_composition_series at (tau=BLEND_DAYS, linear), and that both kernels
    stay on the simplex. Pure-numpy -- no model, runs in a second."""
    from lng_pinn.composition import build_composition_series

    ts = _get_timeseries()
    idx = ts.index
    ref = build_composition_series(idx, seed=seed).to_numpy()

    arrivals_full = _global_cargo_schedule(seed, len(idx))
    traj_full = build_blended_trajectory(arrivals_full, len(idx), BLEND_DAYS, "linear")
    print(f"[self-check] seed={seed}  n_total={len(idx)}")
    print(f"  full series  max|delta vs build_composition_series| = "
          f"{np.abs(traj_full - ref).max():.2e}  (expect ~1e-16)")

    for year in sorted(set(int(y) for y in idx.year)):
        arr, n = load_arrivals(seed, year)
        for kern in ("linear", "exp"):
            tj = build_blended_trajectory(arr, n, BLEND_DAYS, kern)
            simplex = np.abs(tj.sum(axis=1) - 1.0).max()
            nonneg = float(tj.min())
            if kern == "linear":
                e = np.abs(tj - ref[idx.year == year]).max()
                print(f"  {year} linear: max|delta|={e:.2e}  "
                      f"max|rowsum-1|={simplex:.2e}  min={nonneg:.3f}  n={n}")
            else:
                print(f"  {year} exp:    max|rowsum-1|={simplex:.2e}  "
                      f"min={nonneg:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbon-price", type=float, default=80.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    ap.add_argument("--tau-grid", type=float, nargs="+",
                    default=[1, 2, 3, 5, 7, 10])
    ap.add_argument("--kernels", nargs="+", default=["linear", "exp"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--max-tasks-per-child", type=int, default=8,
        help=(
            "Restart the process pool after each worker has handled about this "
            "many cells. This bounds per-process CoolProp/PINN aux memory without "
            "using ProcessPoolExecutor's internal worker recycling, which can "
            "stall after the first recycle wave on some Python/platform "
            "combinations. 0 keeps one pool for the full run."
        ),
    )
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore the cached cells and recompute every coordinate.")
    ap.add_argument("--self-check", action="store_true",
                    help="Validate the wiring (no model needed) and exit.")
    ap.add_argument("--out", default="results/tables/mixing_sensitivity.csv")
    ap.add_argument(
        "--flush-every", type=int, default=1,
        help=(
            "Write the resumable per-seed-year cache after this many completed "
            "tasks. Default 1 is slower but safer on Windows/long runs."
        ),
    )
    args = ap.parse_args()

    if args.self_check:
        _self_check(args.seeds[0] if args.seeds else 42)
        return

    resume = not args.no_resume
    done = _load_cells_cache() if resume else {}

    tasks = [
        (tau, k, seed, year, args.carbon_price)
        for k in args.kernels
        for tau in args.tau_grid
        for seed in args.seeds
        for year in args.years
        if _cell_key(tau, k, seed, year) not in done
    ]
    n_workers = max(1, min(args.workers, os.cpu_count() or 1))

    # Window total (for the smooth progress bar): windows depend only on the
    # year's hour count, which is shared across seeds/tau/kernel.
    H = HORIZON_DAYS * 24
    ts0 = _get_timeseries()
    win_per_year = {
        int(y): max(0, len(range(0, int((ts0.index.year == y).sum()) - H + 1, 24)))
        for y in args.years
    }
    total_windows = sum(win_per_year.get(int(t[3]), 0) for t in tasks)

    print(
        f"carbon_price={args.carbon_price:.1f} EUR/tCO2  "
        f"grid={len(args.kernels)}x{len(args.tau_grid)}  "
        f"seed-years={len(args.seeds)*len(args.years)}  "
        f"cached={len(done)}  to_compute={len(tasks)} cells "
        f"({total_windows} windows)  workers={n_workers}"
    )

    global _SERIAL_BAR
    flush_every = max(1, int(args.flush_every))
    since = 0
    if tasks and n_workers == 1:
        _SERIAL_BAR = tqdm(total=total_windows, desc="windows", unit="win", position=1)
        for t in tqdm(tasks, desc="cells", unit="cell", position=0):
            res = _saving_task(t)
            done[_cell_key(res[0], res[1], res[2], res[3])] = res[4]
            since += 1
            if since >= flush_every:
                _flush_cells_cache(done)
                since = 0
        _SERIAL_BAR.close()
        _flush_cells_cache(done)
    elif tasks:
        mgr = Manager()
        q = mgr.Queue()
        bar_cells = tqdm(total=len(tasks), desc="cells", unit="cell", position=0)
        bar_win = tqdm(total=total_windows, desc="windows", unit="win", position=1)
        stop = threading.Event()

        def _drain(final: bool = False) -> int:
            n = 0
            try:
                while True:
                    q.get_nowait()
                    n += 1
            except Exception:
                pass
            return n

        def _poll() -> None:
            while not stop.is_set():
                got = _drain()
                if got:
                    bar_win.update(got)
                time.sleep(0.3)

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
        pool_kwargs = dict(max_workers=n_workers, initializer=_init_worker, initargs=(q,))
        if args.max_tasks_per_child > 0:
            batch_size = max(n_workers, n_workers * args.max_tasks_per_child)
        else:
            batch_size = len(tasks)
        broke = False
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_no = i // batch_size + 1
            n_batches = (len(tasks) + batch_size - 1) // batch_size
            bar_cells.set_postfix_str(f"pool {batch_no}/{n_batches}", refresh=False)
            try:
                with ProcessPoolExecutor(**pool_kwargs) as ex:
                    futs = {ex.submit(_saving_task, t): t for t in batch}
                    for f in as_completed(futs):
                        res = f.result()
                        done[_cell_key(res[0], res[1], res[2], res[3])] = res[4]
                        bar_cells.update(1)
                        since += 1
                        if since >= flush_every:
                            _flush_cells_cache(done)
                            since = 0
            except BrokenProcessPool:
                broke = True
                _flush_cells_cache(done)
                break
            _flush_cells_cache(done)
            since = 0
        stop.set()
        poller.join(timeout=2)
        bar_win.update(_drain(final=True))  # flush any stragglers
        bar_cells.close()
        bar_win.close()
        _flush_cells_cache(done)
        if broke:
            print(
                f"\nA worker process died (likely WinError 1455 — paging file "
                f"exhausted). {len(done)} cells are safely cached; just re-run the "
                f"same command to resume. If it recurs, lower --workers or "
                f"--max-tasks-per-child."
            )
            return

    rows = []
    for k in args.kernels:
        for tau in args.tau_grid:
            records = [
                {
                    "seed": seed,
                    "year": year,
                    "saving_pct": done[_cell_key(tau, k, seed, year)],
                }
                for seed in args.seeds
                for year in args.years
                if _cell_key(tau, k, seed, year) in done
            ]
            if records:
                rows.append(_aggregate(tau, k, records))
    df = pd.DataFrame(rows).sort_values(["kernel", "tau_days"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.float_format", lambda v: f"{v:+.2f}")
    print(df.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
