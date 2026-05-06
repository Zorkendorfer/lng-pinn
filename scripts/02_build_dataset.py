"""Build PINN training dataset via Latin-hypercube sampling + CoolProp simulation."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.dataset import build_training_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on N=1000 to verify pipeline quickly")
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    N = 1_000 if args.dry_run else args.n
    print(f"git_sha={git_sha}  N={N}  seed={args.seed}  dry_run={args.dry_run}")

    t0 = time.perf_counter()
    df = build_training_set(N=N, seed=args.seed)
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.1f}s  ({N / elapsed:.0f} samples/s)")
    print(df.describe())

    # Energy balance check: W_total must be positive
    bad = (df["W_total"] <= 0).sum()
    if bad:
        print(f"WARNING: {bad} rows with non-positive W_total — check plant.py")
    else:
        print("Energy balance check passed: all W_total > 0")


if __name__ == "__main__":
    main()
