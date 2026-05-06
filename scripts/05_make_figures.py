"""Generate all paper figures from dispatch results."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.plots import (
    fig_cost_delta,
    fig_load_shift_heatmap,
    fig_sensitivity,
    fig_surrogate_fidelity,
)

RESULTS_DIR   = Path("results/tables")
PROCESSED_DIR = Path("data/processed")


def main() -> None:
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

    fig_cost_delta(aware_df, blind_df)
    print("fig1_cost_delta.pdf written")

    # Sensitivity: vary cargo schedule seed and re-run (stub — uses precomputed column)
    if "variability" in aware_df.columns:
        fig_sensitivity(aware_df)
        print("fig2_sensitivity.pdf written")

    fig_load_shift_heatmap(aware_df, blind_df, ts_df)
    print("fig3_load_shift.pdf written")

    if (RESULTS_DIR / "fidelity.parquet").exists():
        fidelity_df = pd.read_parquet(RESULTS_DIR / "fidelity.parquet")
        fig_surrogate_fidelity(fidelity_df)
        print("fig4_fidelity.pdf written")


if __name__ == "__main__":
    main()
