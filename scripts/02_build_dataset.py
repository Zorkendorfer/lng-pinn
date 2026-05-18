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
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="N=1000, skip timeseries build")
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    N = 1_000 if args.dry_run else args.n
    print(f"git_sha={git_sha}  N={N}  seed={args.seed}  dry_run={args.dry_run}")

    # --- Training set ---
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

    # --- Timeseries (skip in dry-run) ---
    if not args.dry_run:
        print(f"\nBuilding timeseries {args.start} → {args.end} ...")
        t1 = time.perf_counter()
        ts = build_timeseries(start=args.start, end=args.end)
        print(f"Timeseries done in {time.perf_counter() - t1:.1f}s  ({len(ts)} rows)")
        print(f"NaN check: {ts.isna().sum().to_dict()}")
        print(ts.describe())


if __name__ == "__main__":
    main()
