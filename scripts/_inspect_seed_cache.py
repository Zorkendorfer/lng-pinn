"""Throwaway: inspect what's currently in data/processed/seed_*.parquet."""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # so the dashes/arrows render on Windows


def inspect_seed_dispatch(seed: int) -> None:
    path = Path(f"data/processed/seed_sensitivity_seed{seed}.parquet")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    n_per_strat = df["_strategy"].value_counts().to_dict()
    print(f"seed={seed}: {len(df)} rows  per-strategy={n_per_strat}")
    print(f"  time range: {df.time.min()} -- {df.time.max()}")

    pivot = (
        df.assign(year=df.time.dt.year)
        .groupby(["year", "_strategy"])["cost_eur"]
        .sum()
        .unstack()
    )
    print("  PINN-predicted yearly cost_eur (millions):")
    print((pivot / 1e6).round(3).to_string())

    pivot["saving_vs_lagged_pct"]  = (pivot["lagged"]  - pivot["aware"]) / pivot["lagged"]  * 100
    pivot["saving_vs_horizon_pct"] = (pivot["horizon"] - pivot["aware"]) / pivot["horizon"] * 100
    print("  PINN-implied saving (%):")
    print(pivot[["saving_vs_lagged_pct", "saving_vs_horizon_pct"]].round(3).to_string())

    # m_dot distribution
    md = df["m_dot"].describe()
    print(f"  m_dot distribution across all strategies: min={md['min']:.1f} "
          f"25%={md['25%']:.1f} 50%={md['50%']:.1f} 75%={md['75%']:.1f} max={md['max']:.1f}")
    print()


def inspect_true_cost_inprogress() -> None:
    path = Path("data/processed/seed_true_costs_seed42_aware_inprogress.parquet")
    if not path.exists():
        return
    df = pd.read_parquet(path)
    print("=== Phase-2 inprogress (seed=42, aware) ===")
    print(f"  rows: {len(df)} / ~26136 (= {100*len(df)/26136:.1f}% complete)")
    print(f"  columns: {list(df.columns)}")
    print(f"  true_cost_eur stats: mean={df['true_cost_eur'].mean():.0f}  "
          f"min={df['true_cost_eur'].min():.0f}  max={df['true_cost_eur'].max():.0f}")


print("=== seed_sensitivity_seed*.parquet (Phase 1 done) ===\n")
for s in [0, 1, 7, 13, 42]:
    inspect_seed_dispatch(s)

inspect_true_cost_inprogress()
