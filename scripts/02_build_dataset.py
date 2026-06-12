"""Build PINN training dataset and hourly timeseries."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.dataset import build_timeseries, build_training_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--zone", default="LT",
        help="v1.4 B — ENTSO-E price zone. Non-LT zones write "
             "timeseries_<zone>.parquet (LT keeps the bare name). The training "
             "set is zone-independent and is only rebuilt for LT.",
    )
    parser.add_argument("--dry-run", action="store_true", help="N=1000, skip timeseries build")
    parser.add_argument(
        "--timeseries-only",
        action="store_true",
        help="Skip the CoolProp training set and rebuild only data/processed/timeseries.parquet.",
    )
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    N = 1_000 if args.dry_run else args.n
    print(f"git_sha={git_sha}  N={N}  seed={args.seed}  zone={args.zone}  dry_run={args.dry_run}")

    # --- Training set (zone-independent; only build for the default LT run) ---
    if args.timeseries_only:
        print("timeseries-only: skipping training-set rebuild.")
    elif args.zone == "LT":
        t0 = time.perf_counter()
        df = build_training_set(N=N, seed=args.seed, workers=args.workers)
        elapsed = time.perf_counter() - t0
        print(f"Training set done in {elapsed:.1f}s  ({N / elapsed:.0f} samples/s)")
        print(df.describe())

        bad = (df["W_total"] <= 0).sum()
        if bad:
            print(f"WARNING: {bad} rows with non-positive W_total")
        else:
            print("Energy balance check passed: all W_total > 0")
    else:
        print(f"zone={args.zone}: skipping training-set rebuild (it is zone-independent).")

    # --- Timeseries (skip in dry-run) ---
    if not args.dry_run:
        print(f"\nBuilding timeseries {args.start} → {args.end}  zone={args.zone} ...")
        t1 = time.perf_counter()
        ts = build_timeseries(start=args.start, end=args.end, zone=args.zone)
        print(f"Timeseries done in {time.perf_counter() - t1:.1f}s  ({len(ts)} rows)")
        print(f"NaN check: {ts.isna().sum().to_dict()}")
        print(ts.describe())


if __name__ == "__main__":
    main()
