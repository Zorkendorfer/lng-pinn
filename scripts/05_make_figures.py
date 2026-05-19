"""Generate all paper figures from dispatch results."""

import argparse
import subprocess
import sys
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
    fig_surrogate_fidelity,
)

RESULTS_DIR = Path("results/tables")
PROCESSED_DIR = Path("data/processed")
COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]


def build_sensitivity_table(
    aware_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """Summarise dispatch savings by composition-variability window."""
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


def _eval_true_costs(
    dispatch_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Re-evaluate a dispatch schedule through CoolProp to get true costs.

    dispatch_df must have columns: time (index), m_dot, cost_eur.
    Returns df with added true_cost_eur column, indexed by time.
    """
    joined = dispatch_df.join(ts_df, how="inner")
    true_costs = []
    for time, row in tqdm(joined.iterrows(), total=len(joined), desc=f"CoolProp {label}", leave=False):
        composition = tuple(float(row[col]) for col in COMP_COLS)
        try:
            out = simulate(composition, float(row["m_dot"]), float(row["T_amb"]), float(row["T_sw"]))
            true_costs.append(float(row["price_eur_mwh"] * out.W_total * row["m_dot"] * 3600.0 / 1000.0))
        except ValueError:
            true_costs.append(np.nan)
    joined["true_cost_eur"] = true_costs
    return joined[["m_dot", "cost_eur", "true_cost_eur"]].dropna()


def build_true_cost_summary(
    strategy_dfs: dict[str, pd.DataFrame],
    ts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all strategies through CoolProp and build yearly true-cost summary."""
    true_dfs = {}
    for name, df in strategy_dfs.items():
        path = RESULTS_DIR / f"true_costs_{name}.parquet"
        if path.exists():
            true_dfs[name] = pd.read_parquet(path)
        else:
            true_dfs[name] = _eval_true_costs(df, ts_df, name)
            true_dfs[name].to_parquet(path, index=False)

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
) -> pd.DataFrame:
    """Re-evaluate sampled PINN dispatch points through the CoolProp simulator."""
    joined = aware_df.join(ts_df, how="inner")
    if n_samples < len(joined):
        sample_idx = np.linspace(0, len(joined) - 1, n_samples, dtype=int)
        joined = joined.iloc[sample_idx]

    records = []
    for time, row in tqdm(joined.iterrows(), total=len(joined), desc="Fidelity", unit="pts"):
        composition = tuple(float(row[col]) for col in COMP_COLS)
        try:
            out = simulate(composition, float(row["m_dot"]), float(row["T_amb"]), float(row["T_sw"]))
        except ValueError:
            continue
        true_cost = float(row["price_eur_mwh"] * out.W_total * row["m_dot"] * 3600.0 / 1000.0)
        records.append({
            "time": time,
            "m_dot": float(row["m_dot"]),
            "pinn_cost_eur": float(row["cost_eur"]),
            "true_cost_eur": true_cost,
            "abs_error_eur": float(row["cost_eur"] - true_cost),
            "rel_error": float(row["cost_eur"] / true_cost - 1.0) if true_cost else np.nan,
        })

    table = pd.DataFrame(records).dropna()
    table.to_parquet(RESULTS_DIR / "fidelity.parquet", index=False)
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
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"
    print(f"git_sha={git_sha}")

    def _load(name: str) -> pd.DataFrame:
        path = RESULTS_DIR / f"{name}.parquet"
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time")

    aware_df    = _load("dispatch_v1")
    horizon_df  = _load("baseline_horizon_v1") if (RESULTS_DIR / "baseline_horizon_v1.parquet").exists() else _load("baseline_v1")
    lagged_df   = _load("baseline_lagged_v1")  if (RESULTS_DIR / "baseline_lagged_v1.parquet").exists()  else horizon_df.copy()
    annual_df   = _load("baseline_annual_v1")  if (RESULTS_DIR / "baseline_annual_v1.parquet").exists()  else horizon_df.copy()
    constant_df = _load("baseline_constant_v1") if (RESULTS_DIR / "baseline_constant_v1.parquet").exists() else horizon_df.copy()
    blind_df    = horizon_df  # keep for backward-compat with existing figure functions
    ts_df = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts_df.index = pd.to_datetime(ts_df.index, utc=True)

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
    build_true_cost_summary(strategy_dfs, ts_df)
    print("yearly_summary_true.csv written (CoolProp true costs)")

    surrogate_eval_path = RESULTS_DIR / "surrogate_eval.parquet"
    if surrogate_eval_path.exists():
        eval_df = pd.read_parquet(surrogate_eval_path)
        eval_df.to_csv(RESULTS_DIR / "surrogate_eval.csv", index=False)
        fig_surrogate_eval(eval_df)
        print("fig5_surrogate_eval.pdf written")

    fig_cost_delta(aware_df, blind_df)
    print("fig1_cost_delta.pdf written")

    sensitivity_df = build_sensitivity_table(aware_df, blind_df, ts_df, args.horizon_days)
    fig_sensitivity(sensitivity_df)
    print("fig2_sensitivity.pdf written")

    fig_load_shift_heatmap(aware_df, blind_df, ts_df)
    print("fig3_load_shift.pdf written")

    fidelity_df = build_fidelity_table(aware_df, ts_df, args.fidelity_samples)
    fig_surrogate_fidelity(fidelity_df)
    print("fig4_fidelity.pdf written")

    export_csv_tables()


if __name__ == "__main__":
    main()
