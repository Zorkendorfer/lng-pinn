"""Replay saved dispatch schedules into electricity and carbon cost components.

This is a cheap post-processor: it does not rerun dispatch and does not call
CoolProp. It combines the saved hourly schedules, saved true-cost caches, and
the deterministic per-seed composition trajectory to split total realised cost
into electricity and carbon components.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.composition import COMP_COLS, build_composition_series
from lng_pinn.thermo import co2_per_kg_fuel

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
DEFAULT_SEEDS = [42, 0, 1, 7, 13, 19, 23, 31, 37, 53]
DEFAULT_STRATEGIES = ["aware", "lagged", "horizon"]


def _carbon_token(price: float) -> str:
    return f"{price:g}".replace("-", "m").replace(".", "p")


def _run_tag(surrogate: str, carbon_price: float) -> str:
    return f"{surrogate}_co2{_carbon_token(carbon_price)}"


def _strategy_column(strategy: str) -> str:
    if strategy == "aware":
        return "aware_eur"
    return f"blind_{strategy}_eur"


def _seed_schedule_path(tag: str, seed: int) -> Path:
    return PROCESSED_DIR / f"seed_sensitivity_{tag}_seed{seed}.parquet"


def _true_cost_path(tag: str, seed: int, strategy: str) -> Path:
    return PROCESSED_DIR / f"seed_true_costs_{tag}_seed{seed}_{strategy}.parquet"


def _reported_costs_path(tag: str) -> Path:
    return RESULTS_DIR / f"seed_sensitivity_{tag}.csv"


def _load_seed_timeseries(seed: int) -> pd.DataFrame:
    ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts.index = pd.to_datetime(ts.index, utc=True)
    comp = build_composition_series(ts.index, seed=seed)
    for col in COMP_COLS:
        ts[col] = comp[col].to_numpy(dtype=float)
    return ts


def _load_reported_costs(tag: str) -> pd.DataFrame | None:
    path = _reported_costs_path(tag)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.set_index(["seed", "year"])


def _component_rows(
    *,
    tag: str,
    seed: int,
    carbon_price: float,
    strategies: list[str],
    reported: pd.DataFrame | None,
) -> list[dict[str, object]]:
    schedule_path = _seed_schedule_path(tag, seed)
    if not schedule_path.exists():
        raise FileNotFoundError(f"missing {schedule_path}")
    sched_all = pd.read_parquet(schedule_path)
    sched_all["time"] = pd.to_datetime(sched_all["time"], utc=True)

    ts = _load_seed_timeseries(seed)
    ts_comp = ts[COMP_COLS].copy()
    ts_comp["co2_kg_per_kg_fuel"] = [
        co2_per_kg_fuel(tuple(row)) for row in ts_comp[COMP_COLS].to_numpy(dtype=float)
    ]

    rows: list[dict[str, object]] = []
    for strategy in strategies:
        true_path = _true_cost_path(tag, seed, strategy)
        if not true_path.exists():
            raise FileNotFoundError(f"missing {true_path}")

        sched = sched_all[sched_all["_strategy"].astype(str) == strategy].copy()
        true = pd.read_parquet(true_path)
        true["time"] = pd.to_datetime(true["time"], utc=True)

        merged = (
            sched[["time", "m_dot"]]
            .merge(true[["time", "true_cost_eur"]], on="time", how="inner")
            .set_index("time")
            .join(ts_comp, how="left")
        )
        if merged.empty:
            raise ValueError(f"no rows after join for seed={seed}, strategy={strategy}")
        if merged[COMP_COLS + ["co2_kg_per_kg_fuel"]].isna().any().any():
            raise ValueError(f"composition join produced NaNs for seed={seed}, {strategy}")

        merged["year"] = merged.index.year
        merged["carbon_cost_eur"] = (
            float(carbon_price) * merged["co2_kg_per_kg_fuel"] * merged["m_dot"] * 3.6
        )
        merged["electricity_cost_eur"] = merged["true_cost_eur"] - merged["carbon_cost_eur"]
        merged["delivered_mass_kg"] = merged["m_dot"] * 3600.0
        merged["co2_t"] = (
            merged["co2_kg_per_kg_fuel"] * merged["delivered_mass_kg"] / 1000.0
        )

        for year, g in merged.groupby("year", sort=True):
            row: dict[str, object] = {
                "seed": seed,
                "year": int(year),
                "strategy": strategy,
                "n_hours": int(len(g)),
                "total_cost_eur": float(g["true_cost_eur"].sum()),
                "electricity_cost_eur": float(g["electricity_cost_eur"].sum()),
                "carbon_cost_eur": float(g["carbon_cost_eur"].sum()),
                "delivered_mass_kg": float(g["delivered_mass_kg"].sum()),
                "co2_t": float(g["co2_t"].sum()),
            }
            if reported is not None and (seed, int(year)) in reported.index:
                col = _strategy_column(strategy)
                if col in reported.columns:
                    reported_total = float(reported.loc[(seed, int(year)), col])
                    row["reported_total_eur"] = reported_total
                    row["relative_total_error_pct"] = (
                        100.0 * (float(row["total_cost_eur"]) - reported_total) / reported_total
                    )
            rows.append(row)
    return rows


def _delta_table(components: pd.DataFrame, baseline: str) -> pd.DataFrame:
    idx = ["seed", "year"]
    aware = components[components["strategy"] == "aware"].set_index(idx)
    base = components[components["strategy"] == baseline].set_index(idx)
    common = aware.index.intersection(base.index)
    if common.empty:
        raise ValueError(f"no common aware/{baseline} seed-year rows")

    rows = []
    for key in common:
        a = aware.loc[key]
        b = base.loc[key]
        total_saving = float(b["total_cost_eur"] - a["total_cost_eur"])
        carbon_saving = float(b["carbon_cost_eur"] - a["carbon_cost_eur"])
        electricity_saving = float(b["electricity_cost_eur"] - a["electricity_cost_eur"])
        mass_delta_pct = (
            100.0
            * float(a["delivered_mass_kg"] - b["delivered_mass_kg"])
            / float(b["delivered_mass_kg"])
        )
        rows.append(
            {
                "seed": int(key[0]),
                "year": int(key[1]),
                "baseline": baseline,
                "saving_total_eur": total_saving,
                "saving_electricity_eur": electricity_saving,
                "saving_carbon_eur": carbon_saving,
                "carbon_fraction_of_saving": carbon_saving / total_saving
                if not math.isclose(total_saving, 0.0)
                else float("nan"),
                "saving_pct": 100.0 * total_saving / float(b["total_cost_eur"]),
                "delivered_mass_delta_pct": mass_delta_pct,
                "co2_saving_t": float(b["co2_t"] - a["co2_t"]),
                # Volume-matched metrics: compare per-delivered-kg cost and
                # CO2 intensity, removing the realised-volume channel that the
                # rolling demand floor leaves free between strategies.
                "saving_per_kg_pct": 100.0
                * (
                    1.0
                    - (float(a["total_cost_eur"]) / float(a["delivered_mass_kg"]))
                    / (float(b["total_cost_eur"]) / float(b["delivered_mass_kg"]))
                ),
                "co2_intensity_saving_pct": 100.0
                * (
                    1.0
                    - (float(a["co2_t"]) / float(a["delivered_mass_kg"]))
                    / (float(b["co2_t"]) / float(b["delivered_mass_kg"]))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["baseline", "seed", "year"])


def _summary_row(scope: str, g: pd.DataFrame) -> dict[str, object]:
    vals = g["saving_pct"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    mean = float(vals.mean()) if n else float("nan")
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    t_stat = mean / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if n > 1 and se > 0 else float("nan")
    try:
        wilcoxon_p = (
            float(stats.wilcoxon(vals, zero_method="wilcox").pvalue)
            if n > 0 and not np.allclose(vals, 0.0)
            else float("nan")
        )
    except ValueError:
        wilcoxon_p = float("nan")
    total = float(g["saving_total_eur"].sum())
    carbon = float(g["saving_carbon_eur"].sum())
    return {
        "scope": scope,
        "n": n,
        "mean_saving_pct": mean,
        "se_pct": se,
        "ci95_lo_pct": mean - 1.96 * se,
        "ci95_hi_pct": mean + 1.96 * se,
        "t_stat": t_stat,
        "p_two_sided": p,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "mean_carbon_fraction_of_saving": float(g["carbon_fraction_of_saving"].mean()),
        "aggregate_carbon_fraction_of_saving": carbon / total
        if not math.isclose(total, 0.0)
        else float("nan"),
        "mean_delivered_mass_delta_pct": float(g["delivered_mass_delta_pct"].mean()),
        "mean_co2_saving_t": float(g["co2_saving_t"].mean()),
        **_per_kg_stats(g),
    }


def _per_kg_stats(g: pd.DataFrame) -> dict[str, float]:
    """Mean and 95% t-interval for the volume-matched per-kg saving."""
    out: dict[str, float] = {}
    if "saving_per_kg_pct" not in g.columns:
        return out
    vals = g["saving_per_kg_pct"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return out
    mean = float(vals.mean())
    se = float(vals.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    if n > 1 and se > 0:
        ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    else:
        ci_lo, ci_hi = mean, mean
    out["mean_saving_per_kg_pct"] = mean
    out["ci95_lo_per_kg_pct"] = float(ci_lo)
    out["ci95_hi_per_kg_pct"] = float(ci_hi)
    if "co2_intensity_saving_pct" in g.columns:
        co2 = g["co2_intensity_saving_pct"].to_numpy(dtype=float)
        co2 = co2[np.isfinite(co2)]
        if co2.size:
            out["mean_co2_intensity_saving_pct"] = float(co2.mean())
    return out


def _summary_table(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (baseline, year), g in delta.groupby(["baseline", "year"], sort=True):
        row = _summary_row(str(year), g)
        row["baseline"] = baseline
        rows.append(row)

    for baseline, g in delta.groupby("baseline", sort=True):
        seed_first = (
            g.groupby("seed", as_index=False)
            .mean(numeric_only=True)
            .assign(baseline=baseline)
        )
        row = _summary_row("ALL_5yr_seed_mean", seed_first)
        row["baseline"] = baseline
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["baseline", "scope"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate", default="hard")
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--baseline", default="lagged")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_DIR / "cost_decomposition.csv"))
    parser.add_argument(
        "--delta-out",
        default=str(RESULTS_DIR / "cost_decomposition_delta.csv"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(RESULTS_DIR / "cost_decomposition_summary.csv"),
    )
    args = parser.parse_args()

    tag = _run_tag(args.surrogate, args.carbon_price)
    reported = _load_reported_costs(tag)
    rows = []
    for seed in args.seeds:
        rows.extend(
            _component_rows(
                tag=tag,
                seed=seed,
                carbon_price=args.carbon_price,
                strategies=list(args.strategies),
                reported=reported,
            )
        )

    components = pd.DataFrame(rows).sort_values(["strategy", "seed", "year"])
    delta = _delta_table(components, baseline=args.baseline)
    summary = _summary_table(delta)

    rel_err = components.get("relative_total_error_pct", pd.Series(dtype=float))
    max_rel_err = float(rel_err.abs().max())
    if math.isfinite(max_rel_err) and max_rel_err > 0.01:
        msg = f"reported total mismatch: max relative error = {max_rel_err:.4f}%"
        if args.strict:
            raise SystemExit(msg)
        print(f"warning: {msg}")

    for path, df in [
        (Path(args.out), components),
        (Path(args.delta_out), delta),
        (Path(args.summary_out), summary),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"wrote {path}")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
