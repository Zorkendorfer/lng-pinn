"""Assemble the controlled soft-vs-hard surrogate comparison.

Consumes tagged outputs written by scripts/06_seed_sensitivity.py, e.g.

    results/tables/seed_sensitivity_hard_co20.csv
    results/tables/seed_sensitivity_hard_co280.csv
    results/tables/seed_sensitivity_soft_co20.csv
    results/tables/seed_sensitivity_soft_co280.csv

and writes a compact seed-level summary table for the paper rework.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/tables")
PATTERN = re.compile(r"seed_sensitivity_(?P<surrogate>.+)_co2(?P<co2>m?[0-9]+(?:p[0-9]+)?)\.csv$")
BASELINES = ("lagged", "horizon")


def _carbon_from_token(token: str) -> float:
    return float(token.replace("m", "-").replace("p", "."))


def _load_tagged_inputs(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("seed_sensitivity_*_co2*.csv")):
        if path.name.startswith("seed_sensitivity_summary_"):
            continue
        match = PATTERN.match(path.name)
        if not match:
            continue
        df = pd.read_csv(path)
        if "surrogate" not in df.columns:
            df["surrogate"] = match.group("surrogate")
        if "carbon_price_eur_per_t" not in df.columns:
            df["carbon_price_eur_per_t"] = _carbon_from_token(match.group("co2"))
        df["_source_file"] = path.name
        frames.append(df)

    if not frames:
        raise SystemExit(
            "No tagged seed-sensitivity files found. Run scripts/06_seed_sensitivity.py "
            "for hard/soft and carbon prices 0/80 first."
        )
    return pd.concat(frames, ignore_index=True)


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = {"seed", "surrogate", "carbon_price_eur_per_t"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Seed-sensitivity inputs are missing columns: {sorted(missing)}")

    for baseline in BASELINES:
        col = f"saving_vs_{baseline}_pct"
        if col not in df.columns:
            continue
        seed_means = (
            df.groupby(["surrogate", "carbon_price_eur_per_t", "seed"], as_index=False)[col]
            .mean()
        )
        for (surrogate, carbon_price), group in seed_means.groupby(
            ["surrogate", "carbon_price_eur_per_t"], sort=True
        ):
            values = group[col].to_numpy(dtype=float)
            n = int(values.size)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            se = std / np.sqrt(n) if n > 1 else 0.0
            rows.append(
                {
                    "surrogate": surrogate,
                    "carbon_price_eur_per_t": float(carbon_price),
                    "baseline": baseline,
                    "n_seeds": n,
                    "mean_saving_pct": round(mean, 4),
                    "std_pct": round(std, 4),
                    "se_pct": round(se, 4),
                    "ci95_lo_pct": round(mean - 1.96 * se, 4),
                    "ci95_hi_pct": round(mean + 1.96 * se, 4),
                }
            )
    return pd.DataFrame(rows)


def _contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = summary[summary["baseline"] == "lagged"]
    for carbon_price, group in primary.groupby("carbon_price_eur_per_t", sort=True):
        means = dict(zip(group["surrogate"], group["mean_saving_pct"]))
        if "hard" in means and "soft" in means:
            rows.append(
                {
                    "carbon_price_eur_per_t": float(carbon_price),
                    "soft_minus_hard_saving_pct": round(float(means["soft"] - means["hard"]), 4),
                    "hard_mean_saving_pct": float(means["hard"]),
                    "soft_mean_saving_pct": float(means["soft"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out", default=str(RESULTS_DIR / "soft_vs_hard.csv"))
    parser.add_argument(
        "--contrast-out",
        default=str(RESULTS_DIR / "soft_vs_hard_contrast.csv"),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = _load_tagged_inputs(results_dir)
    summary = _summarise(df)
    contrast = _contrast(summary)

    out = Path(args.out)
    contrast_out = Path(args.contrast_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    contrast.to_csv(contrast_out, index=False)

    print(summary.to_string(index=False))
    if not contrast.empty:
        print("\nPrimary lagged-baseline contrast:")
        print(contrast.to_string(index=False))
    print(f"\nwrote {out}")
    print(f"wrote {contrast_out}")

    zero = summary[
        (summary["baseline"] == "lagged")
        & np.isclose(summary["carbon_price_eur_per_t"].astype(float), 0.0)
    ]
    if {"hard", "soft"}.issubset(set(zero["surrogate"])):
        means = dict(zip(zero["surrogate"], zero["mean_saving_pct"]))
        print(
            "\nZero-carbon check: "
            f"hard={means['hard']:.4f}%  soft={means['soft']:.4f}% "
            "(soft should carry the fabricated signal; hard should be near zero)."
        )
    else:
        print("\nZero-carbon check skipped: hard and soft co2=0 tagged files are both required.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
