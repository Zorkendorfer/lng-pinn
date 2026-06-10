"""Assemble the 10-seed carbon-price ensemble from tagged seed runs.

This is the code-only scaffold for rework plan item 4. The expensive work is
still done by repeated calls to scripts/06_seed_sensitivity.py; this script
validates that the expected tagged CSVs exist, pools by seed, and writes the
mean/CI table needed for a carbon sweep with seed uncertainty across the full
axis.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
DEFAULT_PRICES = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 160.0]
DEFAULT_SURROGATES = ["hard"]
BASELINES = ["lagged", "horizon"]
PATTERN = re.compile(
    r"seed_sensitivity_(?P<surrogate>.+)_co2(?P<co2>m?[0-9]+(?:p[0-9]+)?)(?P<extra>_.+)?\.csv$"
)


def _carbon_token(price: float) -> str:
    return f"{price:g}".replace("-", "m").replace(".", "p")


def _carbon_from_token(token: str) -> float:
    return float(token.replace("m", "-").replace("p", "."))


def _load_inputs(results_dir: Path) -> pd.DataFrame:
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
        df["composition_tag"] = (match.group("extra") or "").lstrip("_")
        df["_source_file"] = path.name
        frames.append(df)
    if not frames:
        raise SystemExit(
            "No tagged seed-sensitivity files found. Run scripts/06_seed_sensitivity.py "
            "for the desired carbon prices first."
        )
    return pd.concat(frames, ignore_index=True)


def _missing_commands(
    *,
    results_dir: Path,
    prices: list[float],
    surrogates: list[str],
    workers: int,
    composition_tag: str,
) -> list[str]:
    existing = {p.name for p in results_dir.glob("seed_sensitivity_*_co2*.csv")}
    commands = []
    for surrogate in surrogates:
        for price in prices:
            suffix = f"_{composition_tag}" if composition_tag else ""
            expected = f"seed_sensitivity_{surrogate}_co2{_carbon_token(price)}{suffix}.csv"
            if expected in existing:
                continue
            extra = "" if surrogate == "hard" else " --model-path results/models/pinn_soft.pt"
            commands.append(
                "uv run python scripts/06_seed_sensitivity.py "
                f"--carbon-price {price:g} --surrogate {surrogate}{extra} --workers {workers}"
            )
    return commands


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline in BASELINES:
        col = f"saving_vs_{baseline}_pct"
        if col not in df.columns:
            continue
        seed_level = (
            df.groupby(["surrogate", "carbon_price_eur_per_t", "seed"], as_index=False)[col]
            .mean()
        )
        for (surrogate, price), g in seed_level.groupby(
            ["surrogate", "carbon_price_eur_per_t"], sort=True
        ):
            vals = g[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            if n == 0:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se = std / math.sqrt(n) if n > 1 else 0.0
            rows.append(
                {
                    "surrogate": surrogate,
                    "carbon_price_eur_per_t": float(price),
                    "baseline": baseline,
                    "n_seeds": n,
                    "mean_saving_pct": round(mean, 4),
                    "std_pct": round(std, 4),
                    "se_pct": round(se, 4),
                    "ci95_lo_pct": round(mean - 1.96 * se, 4),
                    "ci95_hi_pct": round(mean + 1.96 * se, 4),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["surrogate", "baseline", "carbon_price_eur_per_t"]
    )


def _plot(summary: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"figure skipped: {exc}")
        return

    styles = {
        "lagged": {
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
            "label": "Hard vs lagged",
            "zorder": 3,
        },
        "horizon": {
            "color": "#d62728",
            "linestyle": "--",
            "marker": "s",
            "label": "Hard vs horizon",
            "zorder": 4,
        },
    }
    baseline_order = ["lagged", "horizon"]

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, gap_ax) = plt.subplots(
        2,
        1,
        figsize=(6.2, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
    )
    grouped = {
        (surrogate, baseline): g
        for (surrogate, baseline), g in summary.groupby(["surrogate", "baseline"], sort=True)
    }
    for baseline in baseline_order:
        g = grouped.get(("hard", baseline))
        if g is None:
            continue
        g = g.sort_values("carbon_price_eur_per_t")
        x = g["carbon_price_eur_per_t"].to_numpy(dtype=float)
        y = g["mean_saving_pct"].to_numpy(dtype=float)
        lo = g["ci95_lo_pct"].to_numpy(dtype=float)
        hi = g["ci95_hi_pct"].to_numpy(dtype=float)
        style = styles[baseline]
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=style["label"],
            zorder=style["zorder"],
        )
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.10, zorder=1)

    lagged = grouped.get(("hard", "lagged"))
    horizon = grouped.get(("hard", "horizon"))
    if lagged is not None and horizon is not None:
        gap = (
            lagged.sort_values("carbon_price_eur_per_t")[
                ["carbon_price_eur_per_t", "mean_saving_pct"]
            ]
            .merge(
                horizon.sort_values("carbon_price_eur_per_t")[
                    ["carbon_price_eur_per_t", "mean_saving_pct"]
                ],
                on="carbon_price_eur_per_t",
                suffixes=("_lagged", "_horizon"),
            )
        )
        x = gap["carbon_price_eur_per_t"].to_numpy(dtype=float)
        y = (
            gap["mean_saving_pct_lagged"].to_numpy(dtype=float)
            - gap["mean_saving_pct_horizon"].to_numpy(dtype=float)
        )
        gap_ax.axhline(0.0, color="0.35", lw=0.8)
        gap_ax.plot(x, y, color="0.2", marker="D", ms=3.5, lw=1.0)
        gap_ax.set_ylabel("Lagged -\nhorizon\n(pp)")
        gap_ax.set_ylim(min(-0.02, float(y.min()) - 0.005), max(0.03, float(y.max()) + 0.005))

    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_ylabel("Aware saving [%]")
    ax.set_title("Seed-ensemble carbon-price sweep")
    ax.legend(frameon=False, fontsize=8)
    gap_ax.set_xlabel("CO2 price [EUR/tCO2]")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--prices", type=float, nargs="+", default=DEFAULT_PRICES)
    parser.add_argument("--surrogates", nargs="+", default=DEFAULT_SURROGATES)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--composition-tag",
        default="",
        help="Optional suffix tag after the carbon token; default empty = synthetic runs only.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--emit-missing-commands", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_DIR / "carbon_ensemble.csv"))
    parser.add_argument("--figure", default=str(FIG_DIR / "fig_carbon_ensemble.pdf"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    missing = _missing_commands(
        results_dir=results_dir,
        prices=[float(p) for p in args.prices],
        surrogates=[str(s) for s in args.surrogates],
        workers=args.workers,
        composition_tag=str(args.composition_tag),
    )
    if missing:
        print("Missing tagged seed-sensitivity runs:")
        for cmd in missing:
            print(f"  {cmd}")
        if args.strict:
            raise SystemExit(2)
    elif args.emit_missing_commands:
        print("All requested tagged seed-sensitivity runs are present.")

    if args.emit_missing_commands and missing:
        return

    df = _load_inputs(results_dir)
    wanted_prices = {float(p) for p in args.prices}
    wanted_surrogates = {str(s) for s in args.surrogates}
    df = df[
        df["carbon_price_eur_per_t"].astype(float).isin(wanted_prices)
        & df["surrogate"].astype(str).isin(wanted_surrogates)
        & (df["composition_tag"].astype(str) == str(args.composition_tag))
    ]
    if df.empty:
        raise SystemExit("No rows matched the requested surrogates/prices.")

    summary = _summarise(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    _plot(summary, Path(args.figure))

    print(summary.to_string(index=False))
    print(f"\nwrote {out}")
    print(f"wrote {args.figure}")
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        print(f"git_sha={git_sha}")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
