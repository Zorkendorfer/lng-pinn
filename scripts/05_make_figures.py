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
    ch4 = ts_df["CH4"].reindex(aware_df.index)
    ch4_std = ch4.resample(horizon).std().rename("variability")
    ch4_range = ch4.resample(horizon).apply(lambda s: s.max() - s.min()).rename("ch4_range")
    table = pd.concat([saving.rename("saving_eur"), ch4_std, ch4_range], axis=1).dropna()
    table = table.reset_index().rename(columns={"time": "start_time", "index": "start_time"})
    table.to_parquet(RESULTS_DIR / "sensitivity.parquet", index=False)
    table.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
    return table


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
            out = simulate(
                composition,
                float(row["m_dot"]),
                float(row["T_amb"]),
                float(row["T_sw"]),
            )
        except ValueError:
            continue
        true_cost = float(row["price_eur_mwh"] * out.W_total * row["m_dot"] * 3600.0 / 1000.0)
        records.append(
            {
                "time": time,
                "m_dot": float(row["m_dot"]),
                "pinn_cost_eur": float(row["cost_eur"]),
                "true_cost_eur": true_cost,
                "abs_error_eur": float(row["cost_eur"] - true_cost),
                "rel_error": float(row["cost_eur"] / true_cost - 1.0) if true_cost else np.nan,
            }
        )

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

    aware_df = pd.read_parquet(RESULTS_DIR / "dispatch_v1.parquet")
    blind_df  = pd.read_parquet(RESULTS_DIR / "baseline_v1.parquet")
    ts_df     = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")

    aware_df["time"] = pd.to_datetime(aware_df["time"], utc=True)
    blind_df["time"]  = pd.to_datetime(blind_df["time"], utc=True)
    aware_df = aware_df.set_index("time")
    blind_df  = blind_df.set_index("time")
    ts_df.index = pd.to_datetime(ts_df.index, utc=True)

    export_csv_tables()

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
