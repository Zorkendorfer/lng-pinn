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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.baseline import (
    optimize_blind_annual,
    optimize_blind_horizon,
    optimize_blind_lagged,
    optimize_constant_flow,
)
from lng_pinn.composition import CARGO_CYCLE_DAYS
from lng_pinn.dispatch import M_DOT_MAX, optimize
from lng_pinn.pinn import load
from lng_pinn.plant import simulate
from lng_pinn.plots import fig_carbon_sweep
from lng_pinn.thermo import co2_per_kg_fuel

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]
DEFAULT_PRICES = (0.0, 20.0, 40.0, 80.0, 120.0, 160.0)
HORIZON_DAYS = 7
INV_INITIAL = 0.85
STRATEGIES = ("aware", "horizon", "lagged", "annual", "constant")


def _run_dispatch_for_price(
    ts: pd.DataFrame,
    model: object,
    scaler: object,
    carbon_price: float,
) -> dict[str, pd.DataFrame]:
    """Run all 5 strategies on the full timeseries at one carbon price.

    Returns dict[strategy -> DataFrame with columns m_dot, cost_eur,
    indexed by time]. The returned cost_eur is the PINN's prediction
    (electricity + carbon term); true-cost evaluation happens in a
    second pass via CoolProp.
    """
    H = HORIZON_DAYS * 24
    step = 24
    cargo_cycle_hours = CARGO_CYCLE_DAYS * 24
    cargo_amount = 0.55  # fraction of TANK_CAP per cargo — matches 04_run_dispatch.py

    starts = list(range(0, len(ts) - H + 1, step))
    demand_kg = M_DOT_MAX * 0.6 * H * 3600
    annual_composition = ts[COMP_COLS].mean()

    records: dict[str, list[dict]] = {s: [] for s in STRATEGIES}
    inv = {s: INV_INITIAL for s in STRATEGIES}

    pbar = tqdm(starts, desc=f"  co2={carbon_price:.0f}", unit="day", leave=False)
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

    return {s: pd.DataFrame(records[s]).set_index("time") for s in STRATEGIES}


def _true_cost_for_strategy(
    dispatch_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    carbon_price: float,
) -> pd.Series:
    """CoolProp ground-truth cost (electricity + carbon) per hour."""
    joined = dispatch_df.join(ts_df, how="inner")
    out: list[float] = []
    for row in joined.itertuples():
        comp = tuple(float(getattr(row, c)) for c in COMP_COLS)
        try:
            res = simulate(comp, float(row.m_dot), float(row.T_amb), float(row.T_sw))
        except ValueError:
            out.append(np.nan)
            continue
        elec = float(row.price_eur_mwh) * res.W_total * float(row.m_dot) * 3.6
        carbon = carbon_price * co2_per_kg_fuel(comp) * float(row.m_dot) * 3.6
        out.append(elec + carbon)
    return pd.Series(out, index=joined.index, name="true_cost_eur")


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
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}  prices={args.prices}")

    model, scaler = load()
    model.eval()
    ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts.index = pd.to_datetime(ts.index, utc=True)

    all_rows: list[pd.DataFrame] = []
    for price in args.prices:
        cache_path = RESULTS_DIR / f"carbon_sweep_co2_{int(price)}.csv"
        if cache_path.exists() and not args.no_resume:
            print(f"  co2={price:.0f}: using cached {cache_path.name}")
            yearly = pd.read_csv(cache_path)
        else:
            print(f"  co2={price:.0f}: running full backtest...")
            scheds = _run_dispatch_for_price(ts, model, scaler, price)
            true_costs = {s: _true_cost_for_strategy(scheds[s], ts, price) for s in STRATEGIES}
            yearly = _yearly_savings(true_costs)
            yearly["price_co2_eur_per_t"] = price
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            yearly.to_csv(cache_path, index=False)
        if "price_co2_eur_per_t" not in yearly.columns:
            yearly["price_co2_eur_per_t"] = price
        all_rows.append(yearly)

    sweep_df = pd.concat(all_rows, ignore_index=True)
    sweep_df.to_csv(RESULTS_DIR / "carbon_sweep.csv", index=False)
    fig_carbon_sweep(sweep_df)
    print("Saved results/figures/fig6_carbon_sweep.pdf and results/tables/carbon_sweep.csv")


if __name__ == "__main__":
    main()
