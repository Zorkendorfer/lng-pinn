"""Generate all paper figures from dispatch results."""

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

from lng_pinn.plant import simulate
from lng_pinn.plots import (
    fig_cost_delta,
    fig_load_shift_heatmap,
    fig_sensitivity,
    fig_surrogate_eval,
    fig_volatility_vs_saving,
)

RESULTS_DIR = Path("results/tables")
PROCESSED_DIR = Path("data/processed")
COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]


def build_sensitivity_table(
    aware_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    horizon_days: int,
    resume: bool = True,
) -> pd.DataFrame:
    """Summarise dispatch savings by composition-variability window.

    Pure pandas aggregation — fast — but cached so that re-running the figure script
    after only a downstream edit (e.g. plot styling) skips the re-aggregation.
    """
    cache_path = RESULTS_DIR / "sensitivity.parquet"
    if resume and cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as exc:
            print(f"  Could not read cached sensitivity.parquet: {exc}; recomputing")
    horizon = f"{horizon_days}D"
    saving = (blind_df["cost_eur"] - aware_df["cost_eur"]).resample(horizon).sum()
    aligned = ts_df[["price_eur_mwh", "CH4"]].reindex(aware_df.index)
    grouped = aligned.resample(horizon)
    ch4_std = grouped["CH4"].std().rename("variability")
    ch4_range = grouped["CH4"].apply(lambda s: s.max() - s.min()).rename("ch4_range")
    price_volatility = grouped["price_eur_mwh"].std().rename("price_volatility")
    price_ch4_corr = grouped.apply(
        lambda df: df["price_eur_mwh"].corr(df["CH4"])
        if df["price_eur_mwh"].nunique() > 1 and df["CH4"].nunique() > 1
        else 0.0
    ).rename("price_ch4_corr")
    table = pd.concat(
        [
            saving.rename("saving_eur"),
            ch4_std,
            ch4_range,
            price_volatility,
            price_ch4_corr,
        ],
        axis=1,
    ).dropna()
    table = table.reset_index().rename(columns={"time": "start_time", "index": "start_time"})
    table.to_parquet(RESULTS_DIR / "sensitivity.parquet", index=False)
    table.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
    return table


def build_yearly_summary(
    aware_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    constant_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the side-by-side yearly cost table required by v1.1."""
    yearly = pd.DataFrame(
        {
            "aware_eur": aware_df["cost_eur"].resample("YE").sum(),
            "blind_horizon_eur": horizon_df["cost_eur"].resample("YE").sum(),
            "blind_annual_eur": annual_df["cost_eur"].resample("YE").sum(),
            "constant_eur": constant_df["cost_eur"].resample("YE").sum(),
        }
    ).dropna()
    yearly["saving_vs_horizon_pct"] = (
        (yearly["blind_horizon_eur"] - yearly["aware_eur"]) / yearly["blind_horizon_eur"] * 100
    )
    yearly["saving_vs_annual_pct"] = (
        (yearly["blind_annual_eur"] - yearly["aware_eur"]) / yearly["blind_annual_eur"] * 100
    )
    yearly["saving_vs_constant_pct"] = (
        (yearly["constant_eur"] - yearly["aware_eur"]) / yearly["constant_eur"] * 100
    )
    yearly = yearly.reset_index().rename(columns={"time": "year", "index": "year"})
    yearly["year"] = pd.to_datetime(yearly["year"], utc=True).dt.year
    yearly.to_csv(RESULTS_DIR / "yearly_summary.csv", index=False)
    return yearly


def _carbon_token(price: float) -> str:
    return f"{price:g}".replace("-", "m").replace(".", "p")


def build_volatility_table(
    aware_df: pd.DataFrame | None,
    lagged_df: pd.DataFrame | None,
    ts_df: pd.DataFrame,
    horizon_days: int,
    carbon_price: float,
) -> pd.DataFrame:
    """Build the volatility-vs-saving table for Fig. 8.

    Prefer the five-year default-seed true-cost cache when it exists. The generic
    dispatch cache currently covers only the shorter default-dispatch artifact,
    so using the seed cache keeps Fig. 8 aligned with the 2021--2025 paper
    horizon while preserving the original per-window diagnostic shape.
    """
    token = _carbon_token(carbon_price)
    seed = 42
    aware_path = PROCESSED_DIR / f"seed_true_costs_hard_co2{token}_seed{seed}_aware.parquet"
    lagged_path = PROCESSED_DIR / f"seed_true_costs_hard_co2{token}_seed{seed}_lagged.parquet"
    if aware_path.exists() and lagged_path.exists():
        aware_tc = pd.read_parquet(aware_path)
        lagged_tc = pd.read_parquet(lagged_path)
        aware_tc["time"] = pd.to_datetime(aware_tc["time"], utc=True)
        lagged_tc["time"] = pd.to_datetime(lagged_tc["time"], utc=True)
        aware_tc = aware_tc.set_index("time")["true_cost_eur"]
        lagged_tc = lagged_tc.set_index("time")["true_cost_eur"]
        common = aware_tc.index.intersection(lagged_tc.index)
        horizon = f"{horizon_days}D"
        saving = (lagged_tc.loc[common] - aware_tc.loc[common]).resample(horizon).sum()
        lagged_cost = lagged_tc.loc[common].resample(horizon).sum()
        price_vol = ts_df["price_eur_mwh"].resample(horizon).std()
        diag_df = pd.concat(
            [
                price_vol.rename("price_volatility"),
                saving.rename("saving_eur"),
                (100.0 * saving / lagged_cost).rename("saving_pct"),
            ],
            axis=1,
        ).dropna()
        diag_df = diag_df.reset_index().rename(
            columns={"index": "start_time", "time": "start_time"}
        )
        diag_df["year"] = pd.to_datetime(diag_df["start_time"], utc=True).dt.year
        diag_df.to_csv(RESULTS_DIR / "volatility_vs_saving.csv", index=False)
        return diag_df

    if aware_df is None or lagged_df is None:
        raise SystemExit(
            "Fig. 8 requires either the five-year seed true-cost cache or loaded "
            "dispatch/baseline tables. Run the hard co2=80 seed sensitivity first."
        )

    horizon = f"{horizon_days}D"
    diag_saving = (lagged_df["cost_eur"] - aware_df["cost_eur"]).resample(horizon).sum()
    diag_vol = ts_df["price_eur_mwh"].reindex(aware_df.index).resample(horizon).std()
    diag_df = pd.DataFrame({
        "price_volatility": diag_vol,
        "saving_eur": diag_saving,
    }).dropna()
    diag_df["year"] = diag_df.index.year
    diag_df.to_csv(RESULTS_DIR / "volatility_vs_saving.csv", index=False)
    return diag_df


def _eval_one_row(args: tuple) -> float | None:
    """Top-level helper so ProcessPoolExecutor can pickle it.

    Returns the true total cost per hour: electricity + carbon. The 6-tuple
    form (with carbon_price) is the v1.3 default; older 5-tuples (no carbon)
    are still accepted for cache compatibility.
    """
    if len(args) == 6:
        composition, m_dot, T_amb, T_sw, price, carbon_price = args
    else:
        composition, m_dot, T_amb, T_sw, price = args
        carbon_price = 0.0
    try:
        out = simulate(composition, m_dot, T_amb, T_sw)
    except ValueError:
        return None
    electricity = price * out.W_total * m_dot * 3.6
    if carbon_price > 0.0:
        # Local import keeps the worker pickling cheap.
        from lng_pinn.thermo import co2_per_kg_fuel
        carbon = carbon_price * co2_per_kg_fuel(composition) * m_dot * 3.6
        return float(electricity + carbon)
    return float(electricity)


def _flush_eval_partial(done: dict[int, float], path: Path) -> None:
    """Atomic write of completed (row_idx, true_cost_eur) pairs to a parquet partial."""
    if not done:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"_row_idx": list(done.keys()), "true_cost_eur": list(done.values())}
    )
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _eval_true_costs(
    dispatch_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    label: str,
    resume: bool = True,
    ckpt_every: int = 2000,
    carbon_price_eur_per_t: float = 0.0,
) -> pd.DataFrame:
    """Re-evaluate a dispatch schedule through CoolProp to get true costs.

    Resume behaviour: partial per-row results are flushed every `ckpt_every`
    completions to data/processed/true_costs_partial_<label>.parquet, keyed by
    positional row index (deterministic from the joined frame). On rerun, the
    partial is loaded and only the missing rows are submitted to the pool.
    Removed once the final results/tables/true_costs_<label>.parquet is written.
    """
    joined = dispatch_df.join(ts_df, how="inner")
    rows = list(joined.itertuples())
    n = len(rows)
    all_args: list[tuple[int, tuple]] = []
    for i, row in enumerate(rows):
        all_args.append(
            (
                i,
                (
                    tuple(float(getattr(row, c)) for c in COMP_COLS),
                    float(row.m_dot),
                    float(row.T_amb),
                    float(row.T_sw),
                    float(row.price_eur_mwh),
                    float(carbon_price_eur_per_t),
                ),
            )
        )

    partial_path = PROCESSED_DIR / f"true_costs_partial_{label}.parquet"
    done: dict[int, float] = {}
    if resume and partial_path.exists():
        try:
            prior = pd.read_parquet(partial_path)
            for r in prior.itertuples(index=False):
                done[int(r._row_idx)] = float(r.true_cost_eur)
            print(f"    {label}: resuming with {len(done)}/{n} rows already evaluated")
        except Exception as exc:
            print(f"    {label}: could not resume partial ({exc}); recomputing")
            done = {}

    pending = [a for a in all_args if a[0] not in done]
    if pending:
        n_workers = max(1, os.cpu_count() or 1)
        completed_since_ckpt = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_eval_one_row, a[1]): a[0] for a in pending}
            for future in tqdm(
                as_completed(futures),
                total=len(pending),
                desc=f"CoolProp {label}",
                leave=False,
            ):
                idx = futures[future]
                cost = future.result()
                if cost is not None:
                    done[idx] = cost
                completed_since_ckpt += 1
                if completed_since_ckpt >= ckpt_every:
                    _flush_eval_partial(done, partial_path)
                    completed_since_ckpt = 0
        _flush_eval_partial(done, partial_path)

    joined["true_cost_eur"] = [done.get(i, np.nan) for i in range(n)]
    result = joined[["m_dot", "cost_eur", "true_cost_eur"]].dropna()
    result.index.name = "time"

    if partial_path.exists():
        partial_path.unlink()

    return result


def _restore_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Restore a DatetimeIndex on a cached true-cost frame regardless of how it was saved."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if "time" in df.columns:
        df = df.set_index("time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def build_true_cost_summary(
    strategy_dfs: dict[str, pd.DataFrame],
    ts_df: pd.DataFrame,
    resume: bool = True,
    carbon_price_eur_per_t: float = 0.0,
) -> pd.DataFrame:
    """Evaluate all strategies through CoolProp and build yearly true-cost summary.

    ``carbon_price_eur_per_t`` must match the value dispatch was run at, else
    the true-cost numbers will not be comparable to the PINN-cost numbers
    (which include the carbon term whenever the dispatch was run with it).
    """
    true_dfs = {}
    for name, df in strategy_dfs.items():
        path = RESULTS_DIR / f"true_costs_{name}.parquet"
        if resume and path.exists():
            true_dfs[name] = _restore_time_index(pd.read_parquet(path))
        else:
            true_dfs[name] = _eval_true_costs(
                df, ts_df, name, resume=resume,
                carbon_price_eur_per_t=carbon_price_eur_per_t,
            )
            # Write with `time` as a regular column so the cache round-trips correctly
            # (parquet's `index=False` drops the DatetimeIndex; explicit column survives).
            true_dfs[name].reset_index().to_parquet(path, index=False)

    yearly = pd.DataFrame({
        name: true_dfs[name]["true_cost_eur"].resample("YE").sum()
        for name in strategy_dfs
    }).dropna()

    # Savings vs each baseline
    for baseline in ["lagged", "horizon", "constant"]:
        if baseline in yearly.columns:
            yearly[f"saving_vs_{baseline}_pct"] = (
                (yearly[baseline] - yearly["aware"]) / yearly[baseline] * 100
            )

    yearly = yearly.reset_index()
    yearly["year"] = pd.to_datetime(yearly["time"], utc=True).dt.year
    yearly = yearly.drop(columns=["time"])
    yearly.to_csv(RESULTS_DIR / "yearly_summary_true.csv", index=False)
    return yearly


def build_fidelity_table(
    aware_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    n_samples: int,
    resume: bool = True,
    carbon_price_eur_per_t: float = 0.0,
) -> pd.DataFrame:
    """Re-evaluate sampled PINN dispatch points through the CoolProp simulator.

    Resume-aware: if `results/tables/fidelity.parquet` already exists and matches the
    requested `n_samples`, it's returned without re-running CoolProp.

    ``carbon_price_eur_per_t`` is added to the true-cost computation so the
    PINN cost (which already includes carbon if dispatch was run with it) and
    the true cost are on the same accounting basis.
    """
    final_path = RESULTS_DIR / "fidelity.parquet"
    if resume and final_path.exists():
        try:
            cached = pd.read_parquet(final_path)
            if len(cached) == n_samples or (n_samples >= len(aware_df.join(ts_df, how="inner"))):
                print(f"  Reusing cached fidelity.parquet ({len(cached)} rows)")
                return cached
        except Exception as exc:
            print(f"  Could not read cached fidelity.parquet: {exc}; recomputing")

    # Local import keeps the worker pool unaffected when carbon_price == 0.
    from lng_pinn.thermo import co2_per_kg_fuel

    joined = aware_df.join(ts_df, how="inner")
    if n_samples < len(joined):
        sample_idx = np.linspace(0, len(joined) - 1, n_samples, dtype=int)
        joined = joined.iloc[sample_idx]

    records = []
    for time, row in tqdm(joined.iterrows(), total=len(joined), desc="Fidelity", unit="pts"):
        composition = tuple(float(row[col]) for col in COMP_COLS)
        try:
            out = simulate(
                composition, float(row["m_dot"]), float(row["T_amb"]), float(row["T_sw"])
            )
        except ValueError:
            continue
        electricity = float(row["price_eur_mwh"] * out.W_total * row["m_dot"] * 3.6)
        if carbon_price_eur_per_t > 0.0:
            carbon = float(
                carbon_price_eur_per_t * co2_per_kg_fuel(composition) * row["m_dot"] * 3.6
            )
        else:
            carbon = 0.0
        true_cost = electricity + carbon
        records.append({
            "time": time,
            "m_dot": float(row["m_dot"]),
            "pinn_cost_eur": float(row["cost_eur"]),
            "true_cost_eur": true_cost,
            "abs_error_eur": float(row["cost_eur"] - true_cost),
            "rel_error": float(row["cost_eur"] / true_cost - 1.0) if true_cost else np.nan,
        })

    table = pd.DataFrame(records).dropna()
    table.to_parquet(final_path, index=False)
    table.to_csv(RESULTS_DIR / "fidelity.csv", index=False)
    return table


def export_csv_tables() -> None:
    """Mirror parquet result tables to CSV for paper/review workflows."""
    for parquet_path in RESULTS_DIR.glob("*.parquet"):
        df = pd.read_parquet(parquet_path)
        df.to_csv(parquet_path.with_suffix(".csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--fidelity-samples", type=int, default=1000)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing true_costs / sensitivity / fidelity caches and recompute.",
    )
    parser.add_argument(
        "--carbon-price", type=float, default=0.0,
        help="v1.3 B1: CO2 price (EUR/tCO2) used when re-evaluating dispatch through "
             "CoolProp. Must match the value `04_run_dispatch.py --carbon-price` was "
             "run with, else PINN and true costs won't be comparable. Default 0 "
             "reproduces v1.2 (electricity-only) accounting.",
    )
    parser.add_argument(
        "--only-fig8",
        action="store_true",
        help="Regenerate only volatility_vs_saving.csv and fig8_volatility_vs_saving.pdf.",
    )
    args = parser.parse_args()
    resume = not args.no_resume

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}  carbon_price={args.carbon_price:.1f} EUR/tCO2")
    if args.carbon_price > 0.0 and resume:
        print(
            "  NOTE: --carbon-price > 0 with --resume: existing true_costs_*.parquet "
            "caches may be electricity-only. Delete them or pass --no-resume to rebuild."
        )

    ts_df = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts_df.index = pd.to_datetime(ts_df.index, utc=True)

    if args.only_fig8:
        diag_df = build_volatility_table(
            None, None, ts_df, args.horizon_days, args.carbon_price
        )
        fig_volatility_vs_saving(diag_df)
        print("fig8_volatility_vs_saving.pdf written")
        return

    def _load(name: str) -> pd.DataFrame:
        path = RESULTS_DIR / f"{name}.parquet"
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time")

    aware_df    = _load("dispatch_v1")
    horizon_df = (
        _load("baseline_horizon_v1")
        if (RESULTS_DIR / "baseline_horizon_v1.parquet").exists()
        else _load("baseline_v1")
    )
    lagged_df = (
        _load("baseline_lagged_v1")
        if (RESULTS_DIR / "baseline_lagged_v1.parquet").exists()
        else horizon_df.copy()
    )
    annual_df = (
        _load("baseline_annual_v1")
        if (RESULTS_DIR / "baseline_annual_v1.parquet").exists()
        else horizon_df.copy()
    )
    constant_df = (
        _load("baseline_constant_v1")
        if (RESULTS_DIR / "baseline_constant_v1.parquet").exists()
        else horizon_df.copy()
    )
    blind_df    = horizon_df  # keep for backward-compat with existing figure functions
    export_csv_tables()
    build_yearly_summary(aware_df, blind_df, annual_df, constant_df)
    print("yearly_summary.csv written (PINN costs)")

    # True-cost yearly summary — the honest comparison
    strategy_dfs = {
        "aware":    aware_df,
        "lagged":   lagged_df,
        "horizon":  horizon_df,
        "annual":   annual_df,
        "constant": constant_df,
    }
    build_true_cost_summary(
        strategy_dfs, ts_df,
        resume=resume,
        carbon_price_eur_per_t=args.carbon_price,
    )
    print("yearly_summary_true.csv written (CoolProp true costs)")

    surrogate_eval_path = RESULTS_DIR / "surrogate_eval.parquet"
    if surrogate_eval_path.exists():
        eval_df = pd.read_parquet(surrogate_eval_path)
        eval_df.to_csv(RESULTS_DIR / "surrogate_eval.csv", index=False)
        fig_surrogate_eval(eval_df)
        print("fig5_surrogate_eval.pdf written")

    fig_cost_delta(aware_df, blind_df)
    print("fig1_cost_delta.pdf written")

    sensitivity_df = build_sensitivity_table(
        aware_df, blind_df, ts_df, args.horizon_days, resume=resume
    )
    fig_sensitivity(sensitivity_df)
    print("fig2_sensitivity.pdf written")

    fig_load_shift_heatmap(aware_df, blind_df, ts_df)
    print("fig3_load_shift.pdf written")

    # v1.4 D — price-volatility diagnostic. Prefer the full five-year seed
    # ensemble if available; otherwise fall back to the older per-window cache.
    diag_df = build_volatility_table(
        aware_df, lagged_df, ts_df, args.horizon_days, args.carbon_price
    )
    fig_volatility_vs_saving(diag_df)
    print("fig8_volatility_vs_saving.pdf written")

    # v1.4: fig4 (PINN-vs-CoolProp parity scatter) is dropped from the figure
    # set — the physics-by-construction PINN matches CoolProp to ~1e-7 relative
    # cost error, so the scatter is an uninformative y=x line and fig5's
    # MAE/RMSE/R² bars already document accuracy. We still build fidelity.csv
    # as a numeric backup (it is the quantitative form of the same claim).
    build_fidelity_table(
        aware_df, ts_df, args.fidelity_samples,
        resume=resume,
        carbon_price_eur_per_t=args.carbon_price,
    )
    print("fidelity.csv written (fig4 scatter retired — see fig5 for accuracy)")

    export_csv_tables()


if __name__ == "__main__":
    main()
