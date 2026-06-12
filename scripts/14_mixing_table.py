"""Assemble the tank-mixing robustness table.

This is a lightweight post-processor for the tank-mixing robustness run. It
prefers the per-seed/year cell cache so the paper table is always aggregated
seed-first: each seed is averaged over years before the confidence interval and
tests are computed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path("results/tables")
PROCESSED_DIR = Path("data/processed")
DEFAULT_TAUS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
DEFAULT_KERNELS = ["linear", "exp"]


def _summary_from_cells(cells: pd.DataFrame) -> pd.DataFrame:
    required = {"tau_days", "kernel", "seed", "saving_pct"}
    missing = required - set(cells.columns)
    if missing:
        raise SystemExit(f"cell cache is missing columns: {sorted(missing)}")
    rows = []
    for (kernel, tau), g in cells.groupby(["kernel", "tau_days"], sort=True):
        g = g[np.isfinite(g["saving_pct"].astype(float))]
        vals = g.groupby("seed")["saving_pct"].mean().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        se = std / math.sqrt(n) if n > 1 else 0.0
        t_stat = mean / se if se > 0 else float("nan")
        p = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if n > 1 and se > 0 else float("nan")
        if n > 1 and se > 0:
            ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        else:
            ci_low, ci_high = mean, mean
        try:
            wilcoxon_p = (
                float(stats.wilcoxon(vals, zero_method="wilcox").pvalue)
                if n > 0 and not np.allclose(vals, 0.0)
                else float("nan")
            )
        except ValueError:
            wilcoxon_p = float("nan")
        rows.append(
            {
                "tau_days": float(tau),
                "kernel": str(kernel),
                "n": n,
                "n_seed_year_cells": int(len(g)),
                "mean_pct": mean,
                "std_pct": std,
                "se_pct": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "t": t_stat,
                "p": p,
                "wilcoxon_p_two_sided": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows)


def _load_summary(results_path: Path, cells_path: Path, prefer_summary: bool) -> pd.DataFrame:
    if not prefer_summary and cells_path.exists():
        return _summary_from_cells(pd.read_parquet(cells_path))
    if results_path.exists():
        df = pd.read_csv(results_path)
        if "n_seed_year_cells" not in df.columns and "seed" in df.columns:
            return _summary_from_cells(df)
        return df
    if cells_path.exists():
        return _summary_from_cells(pd.read_parquet(cells_path))
    raise SystemExit(
        f"Neither {results_path} nor {cells_path} exists. "
        "Run scripts/09_mixing_sensitivity.py first."
    )


def _format_cell(row: pd.Series) -> str:
    return f"{row.mean_pct:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]"


def _paper_table(summary: pd.DataFrame, taus: list[float], kernels: list[str]) -> pd.DataFrame:
    rows = []
    for tau in taus:
        row: dict[str, object] = {"tau_days": tau}
        for kernel in kernels:
            hit = summary[
                (summary["kernel"].astype(str) == kernel)
                & np.isclose(summary["tau_days"].astype(float), float(tau))
            ]
            row[f"{kernel}_saving_pct_ci95"] = _format_cell(hit.iloc[0]) if len(hit) else ""
            row[f"{kernel}_n"] = int(hit.iloc[0]["n"]) if len(hit) and "n" in hit.columns else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _missing_cells(
    summary: pd.DataFrame,
    taus: list[float],
    kernels: list[str],
    expected_n: int,
) -> list[str]:
    missing = []
    for kernel in kernels:
        for tau in taus:
            hit = summary[
                (summary["kernel"].astype(str) == kernel)
                & np.isclose(summary["tau_days"].astype(float), float(tau))
            ]
            if hit.empty:
                missing.append(f"{kernel} tau={tau:g}: missing")
                continue
            n = int(hit.iloc[0].get("n", 0))
            if expected_n and n < expected_n:
                missing.append(f"{kernel} tau={tau:g}: n={n}, expected {expected_n}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RESULTS_DIR / "mixing_sensitivity.csv"))
    parser.add_argument(
        "--cells-cache",
        default=str(PROCESSED_DIR / "mixing_sensitivity_cells.parquet"),
    )
    parser.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS)
    parser.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS)
    parser.add_argument("--expected-n", type=int, default=10)
    parser.add_argument(
        "--prefer-summary",
        action="store_true",
        help="Use --input even when the per-seed/year cell cache is available.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_DIR / "mixing_table3.csv"))
    args = parser.parse_args()

    summary = _load_summary(Path(args.input), Path(args.cells_cache), args.prefer_summary)
    missing = _missing_cells(summary, args.taus, args.kernels, args.expected_n)
    if missing:
        print("Incomplete mixing cells:")
        for msg in missing:
            print(f"  {msg}")
        if args.strict:
            raise SystemExit(2)

    table = _paper_table(summary, args.taus, args.kernels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
