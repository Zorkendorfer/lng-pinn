"""v1.4 A — perfect-foresight upper bound and the savings cascade.

Computes the savings cascade  lagged -> aware -> oracle  at a single carbon
price (default 80 EUR/tCO2) on CoolProp ground-truth cost, then writes
results/tables/foresight_cascade.csv and results/figures/fig7_foresight_gap.pdf.

Strategy definitions (all share the same demand basis, cargo schedule, and
initial inventory):
  - lagged : rolling 7-day window, composition frozen at window start (realistic
             operator baseline; the 0% reference for the cascade).
  - aware  : rolling 7-day window, true hourly composition within the window.
  - oracle : non-overlapping blocks (default = one cargo cycle, 12 days) solved
             at once with the entire block's true composition and price — an
             extended/perfect-foresight reference. Blocks are aligned to cargo
             boundaries so deliveries fall *between* blocks (no mid-block cargo),
             keeping every solve on the tested optimize() path.

CORRECTNESS GATE: the script asserts saving(oracle) >= saving(aware) >=
saving(lagged) - eps. If that ordering is violated it prints a loud warning;
that usually means --block-days is <= the rolling lookahead (7) or there is a
dispatch bug. Increase --block-days and rerun.
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

# torch-free imports only at module top so the parallel CoolProp workers don't
# load the CUDA DLLs on spawn (see 06/07 for the same pattern). Dispatch/model
# imports are lazy, inside the functions that need them.
from lng_pinn.composition import CARGO_CYCLE_DAYS

COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
CARGO_CYCLE_HOURS = CARGO_CYCLE_DAYS * 24
CARGO_AMOUNT = 0.55
INV_INITIAL = 0.85
INV_CAP = 0.92
HORIZON_DAYS = 7


def _true_cost_row(args: tuple) -> float | None:
    """Worker: one CoolProp simulation + cost (EUR/h). Mirrors 07's helper."""
    composition, m_dot, T_amb, T_sw, price, carbon_price = args
    from lng_pinn.plant import simulate
    from lng_pinn.thermo import co2_per_kg_fuel

    try:
        res = simulate(composition, m_dot, T_amb, T_sw)
    except ValueError:
        return None
    elec = price * res.W_total * m_dot * 3.6
    carbon = carbon_price * co2_per_kg_fuel(composition) * m_dot * 3.6 if carbon_price > 0 else 0.0
    return float(elec + carbon)


def _rolling_strategy(
    ts: pd.DataFrame, model: object, scaler: object, carbon_price: float,
    demand_factor: float, which: str,
) -> pd.DataFrame:
    """Run aware or lagged as a 7-day rolling, daily-commit backtest.

    Returns a DataFrame indexed by time with columns m_dot, cost_eur (PINN).
    """
    from lng_pinn.baseline import optimize_blind_lagged  # lazy torch import
    from lng_pinn.dispatch import M_DOT_MAX, optimize

    H = HORIZON_DAYS * 24
    step = 24
    starts = list(range(0, len(ts) - H + 1, step))
    demand_kg = M_DOT_MAX * demand_factor * H * 3600
    inv = INV_INITIAL
    records = []
    for start in tqdm(starts, desc=f"  rolling {which}", unit="day", leave=False):
        if start > 0 and start % CARGO_CYCLE_HOURS == 0:
            inv = min(INV_CAP, inv + CARGO_AMOUNT)
        window = ts.iloc[start : start + H]
        n = min(step, len(window))
        if which == "aware":
            sched = optimize(
                window, model, scaler, demand_kg, inv, carbon_price_eur_per_t=carbon_price,
            )
        else:  # lagged
            sched = optimize_blind_lagged(
                window, model, scaler, demand_kg, ts[COMP_COLS].iloc[start], inv,
                carbon_price_eur_per_t=carbon_price,
            )
        for t, row in enumerate(window.iloc[:n].itertuples()):
            records.append({
                "time": row.Index,
                "m_dot": float(sched.m_dot[t]),
                "cost_eur": float(sched.cost_eur[t]),
            })
        inv = float(sched.tank_level[n])
    return pd.DataFrame(records).set_index("time")


def _oracle_strategy(
    ts: pd.DataFrame, model: object, scaler: object, carbon_price: float,
    demand_factor: float, block_hours: int,
) -> pd.DataFrame:
    """Perfect-foresight dispatch over aligned non-overlapping blocks.

    Cargo is injected between blocks (like the rolling drivers inject between
    windows); blocks are full-foresight optimize() solves.
    """
    from lng_pinn.baseline import optimize_perfect_foresight_block  # lazy torch import
    from lng_pinn.dispatch import M_DOT_MAX

    # demand per hour in KG (kg/s * 3600), so block demand = this * n_hours
    # matches the rolling strategies' M_DOT_MAX*demand_factor*H*3600 basis.
    demand_per_hour = M_DOT_MAX * demand_factor * 3600.0
    inv = INV_INITIAL
    records = []
    starts = list(range(0, len(ts), block_hours))
    for bi, start in enumerate(tqdm(starts, desc="  oracle blocks", unit="block", leave=False)):
        if start > 0 and start % CARGO_CYCLE_HOURS == 0:
            inv = min(INV_CAP, inv + CARGO_AMOUNT)
        block = ts.iloc[start : start + block_hours]
        if len(block) == 0:
            continue
        sched = optimize_perfect_foresight_block(
            block, model, scaler, demand_per_hour, inv,
            carbon_price_eur_per_t=carbon_price,
        )
        for t, row in enumerate(block.itertuples()):
            records.append({
                "time": row.Index,
                "m_dot": float(sched.m_dot[t]),
                "cost_eur": float(sched.cost_eur[t]),
            })
        inv = float(sched.tank_level[len(block)])
        _ = bi
    return pd.DataFrame(records).set_index("time")


def _true_cost(
    dispatch_df: pd.DataFrame, ts: pd.DataFrame, carbon_price: float,
    label: str, n_workers: int,
) -> pd.Series:
    """CoolProp ground-truth cost per hour for a dispatch schedule."""
    joined = dispatch_df.join(ts, how="inner")
    args = [
        (
            tuple(float(getattr(r, c)) for c in COMP_COLS),
            float(r.m_dot), float(r.T_amb), float(r.T_sw),
            float(r.price_eur_mwh), float(carbon_price),
        )
        for r in joined.itertuples()
    ]
    out: list[float | None] = [None] * len(args)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_true_cost_row, a): i for i, a in enumerate(args)}
        for fut in tqdm(as_completed(futs), total=len(args), desc=f"  true-cost {label}",
                        unit="hr", leave=False):
            out[futs[fut]] = fut.result()
    return pd.Series([v if v is not None else np.nan for v in out],
                     index=joined.index, name="true_cost_eur").dropna()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument(
        "--block-days", type=int, default=CARGO_CYCLE_DAYS,
        help=f"Oracle block length in days (default {CARGO_CYCLE_DAYS} = one cargo cycle). "
             "Must exceed the 7-day rolling lookahead for a valid upper bound.",
    )
    parser.add_argument("--demand-factor", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--timeseries", default=str(PROCESSED_DIR / "timeseries.parquet"),
        help="v1.4 B — timeseries parquet path (timeseries_<zone>.parquet for a second zone).",
    )
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    block_hours = args.block_days * 24
    print(
        f"git_sha={git_sha}  carbon_price={args.carbon_price:.0f}  "
        f"block_days={args.block_days}  demand_factor={args.demand_factor}"
    )
    if args.block_days <= HORIZON_DAYS:
        print(
            f"  WARNING: --block-days ({args.block_days}) <= rolling lookahead "
            f"({HORIZON_DAYS}); the oracle may not dominate aware. Increase it."
        )

    from lng_pinn.pinn import load  # lazy torch import — see module-top comment
    model, scaler = load()
    model.eval()
    ts = pd.read_parquet(args.timeseries)
    ts.index = pd.to_datetime(ts.index, utc=True)

    print("Phase 1/2 — dispatch (lagged, aware, oracle)...")
    lagged = _rolling_strategy(ts, model, scaler, args.carbon_price, args.demand_factor, "lagged")
    aware = _rolling_strategy(ts, model, scaler, args.carbon_price, args.demand_factor, "aware")
    oracle = _oracle_strategy(
        ts, model, scaler, args.carbon_price, args.demand_factor, block_hours,
    )

    print("Phase 2/2 — CoolProp true-cost re-evaluation...")
    tc = {
        "lagged": _true_cost(lagged, ts, args.carbon_price, "lagged", args.workers),
        "aware": _true_cost(aware, ts, args.carbon_price, "aware", args.workers),
        "oracle": _true_cost(oracle, ts, args.carbon_price, "oracle", args.workers),
    }

    # Align all three on their common hours so the comparison is like-for-like.
    common = tc["lagged"].index.intersection(tc["aware"].index).intersection(tc["oracle"].index)
    rows = []
    for strat in ("lagged", "aware", "oracle"):
        s = tc[strat].loc[common]
        s_year = s.groupby(s.index.year).sum()
        for year, total in s_year.items():
            rows.append({"strategy": strat, "year": int(year), "true_cost_eur": float(total)})
    cascade = pd.DataFrame(rows)

    # Saving vs lagged, per year.
    pivot = cascade.pivot(index="year", columns="strategy", values="true_cost_eur")
    for strat in ("lagged", "aware", "oracle"):
        cascade.loc[cascade["strategy"] == strat, "saving_vs_lagged_pct"] = cascade[
            cascade["strategy"] == strat
        ].apply(
            lambda r: (pivot.loc[r["year"], "lagged"] - r["true_cost_eur"])
            / pivot.loc[r["year"], "lagged"] * 100,
            axis=1,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cascade.to_csv(RESULTS_DIR / "foresight_cascade.csv", index=False)

    # Pooled (all-year) saving and the headline "aware captures X% of ceiling".
    tot = pivot.sum(axis=0)
    save_aware = (tot["lagged"] - tot["aware"]) / tot["lagged"] * 100
    save_oracle = (tot["lagged"] - tot["oracle"]) / tot["lagged"] * 100
    capture = (save_aware / save_oracle * 100) if save_oracle != 0 else float("nan")
    print(f"\n  Pooled saving vs lagged:  aware={save_aware:+.2f}%   oracle={save_oracle:+.2f}%")
    print(f"  Aware captures {capture:.0f}% of the perfect-foresight ceiling.")

    # Correctness gate.
    eps = 0.05  # percentage points of slack for solver noise
    ok = (save_oracle + eps >= save_aware) and (save_aware + eps >= 0.0)
    if not ok:
        print(
            "\n  *** ORDERING VIOLATED: expected oracle >= aware >= 0. "
            "Increase --block-days or check the dispatch. ***"
        )
    else:
        print("  Ordering OK: oracle >= aware >= lagged(0).")

    try:
        from lng_pinn.plots import fig_foresight_gap
        fig_foresight_gap(cascade)
        print("  Saved results/figures/fig7_foresight_gap.pdf and foresight_cascade.csv")
    except Exception as exc:
        print(f"  Figure step skipped ({exc}); foresight_cascade.csv is written.")


if __name__ == "__main__":
    main()
