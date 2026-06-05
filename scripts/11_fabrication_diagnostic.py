#!/usr/bin/env python3
"""
11_fabrication_diagnostic.py  (rework plan item 3)

Runs the composition-fabrication diagnostic (lng_pinn.diagnostics) over the
rolling-horizon backtest windows for a given surrogate and reports, from code,
whether the surrogate invents composition-dependent cost the CoolProp simulator
does not have. A faithful hard-physics surrogate PASSES (gap ~ 0); a soft-physics
surrogate is expected to FAIL.

    # hard surrogate (current checkpoint)
    uv run python scripts/11_fabrication_diagnostic.py --surrogate hard

    # soft surrogate, once scripts/03 can train one (rework plan item 2)
    uv run python scripts/11_fabrication_diagnostic.py --surrogate soft \
        --model-path results/models/pinn_soft.pt

Writes results/tables/fabrication_diagnostic.csv (one summary row per run) plus a
per-window detail CSV and a scatter figure.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # tiny forward passes; CoolProp dominates

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.composition import build_composition_series  # torch-free
from lng_pinn.diagnostics import COMP_COLS, composition_fabrication_gap

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
HORIZON_DAYS = 7
DEMAND_FACTOR = 0.6


def _ts_for_seed(seed: int) -> pd.DataFrame:
    ts = pd.read_parquet(PROCESSED_DIR / "timeseries.parquet")
    ts.index = pd.to_datetime(ts.index, utc=True)
    comp = build_composition_series(ts.index, seed=seed)
    for col in COMP_COLS:
        ts[col] = comp[col]
    return ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate", default="hard",
                    help="Label recorded in the output (e.g. 'hard' or 'soft').")
    ap.add_argument("--model-path", default=None,
                    help="Checkpoint to load; defaults to the shipped hard model.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--carbon-price", type=float, default=80.0)
    ap.add_argument("--max-windows", type=int, default=60,
                    help="Evenly subsample this many windows (CoolProp cost).")
    ap.add_argument("--threshold-frac", type=float, default=0.02,
                    help="Flag windows whose |gap| exceeds this fraction of truth cost.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "fabrication_diagnostic.csv"))
    args = ap.parse_args()

    from lng_pinn.dispatch import M_DOT_MAX
    from lng_pinn.pinn import load

    model, scaler = load(args.model_path) if args.model_path else load()
    model.eval()
    m_dot = M_DOT_MAX * DEMAND_FACTOR

    ts = _ts_for_seed(args.seed)
    H = HORIZON_DAYS * 24
    starts = list(range(0, len(ts) - H + 1, 24))
    if args.max_windows and len(starts) > args.max_windows:
        idx = np.linspace(0, len(starts) - 1, args.max_windows).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx))]

    rows = []
    from tqdm import tqdm
    for start in tqdm(starts, desc=f"{args.surrogate} windows", unit="win"):
        window = ts.iloc[start:start + H]
        d = composition_fabrication_gap(model, scaler, window, m_dot, args.carbon_price)
        d["start"] = start
        rows.append(d)

    detail = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = RESULTS_DIR / f"fabrication_diagnostic_{args.surrogate}_seed{args.seed}.csv"
    detail.to_csv(detail_path, index=False)

    gap_frac = detail["gap_frac_of_truth_cost"].to_numpy()
    mean_abs_gap_frac = float(np.mean(np.abs(gap_frac)))
    flagged = float(np.mean(np.abs(gap_frac) > args.threshold_frac))
    passed = mean_abs_gap_frac <= args.threshold_frac

    summary = {
        "surrogate": args.surrogate,
        "seed": args.seed,
        "carbon_price_eur_per_t": args.carbon_price,
        "n_windows": len(detail),
        "mean_abs_gap_frac": round(mean_abs_gap_frac, 6),
        "median_gap_frac": round(float(np.median(gap_frac)), 6),
        "frac_windows_flagged": round(flagged, 4),
        "mean_surrogate_delta_eur": round(float(detail["surrogate_delta_eur"].mean()), 2),
        "mean_truth_delta_eur": round(float(detail["truth_delta_eur"].mean()), 2),
        "threshold_frac": args.threshold_frac,
        "passed": bool(passed),
    }
    out = Path(args.out)
    header = not out.exists()
    pd.DataFrame([summary]).to_csv(out, mode="a", index=False, header=header)

    # Scatter: truth vs surrogate composition-sensitivity per window. A faithful
    # surrogate hugs the y=x line; a fabricating one departs from it.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        td = detail["truth_delta_eur"]
        sd = detail["surrogate_delta_eur"]
        lim = float(max(td.abs().max(), sd.abs().max(), 1.0))
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1, ls="--", label="faithful (y=x)")
        ax.scatter(td, sd, s=18, color="steelblue", alpha=0.7)
        ax.set_xlabel("CoolProp truth: cost(varying) - cost(mean)  [EUR/window]")
        ax.set_ylabel("Surrogate: cost(varying) - cost(mean)  [EUR/window]")
        ax.set_title(
            f"Composition fabrication: {args.surrogate} surrogate\n"
            f"mean |gap| = {mean_abs_gap_frac*100:.3f}% of truth cost "
            f"({'PASS' if passed else 'FAIL'})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig_fabrication_{args.surrogate}.pdf")
        plt.close(fig)
    except Exception as exc:  # plotting is optional
        print(f"  (figure skipped: {exc})")

    print("\n" + "  ".join(f"{k}={v}" for k, v in summary.items()))
    print(f"wrote {detail_path}")
    print(f"appended summary to {out}")
    print(f"DIAGNOSTIC: {args.surrogate} surrogate "
          f"{'PASSES' if passed else 'FAILS'} at threshold {args.threshold_frac:.0%}")


if __name__ == "__main__":
    main()
