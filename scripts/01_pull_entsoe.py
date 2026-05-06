"""Pull Lithuanian day-ahead prices from ENTSO-E and cache to parquet."""

import argparse
import subprocess
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.market import pull_da_prices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default="2024-01-01")
    parser.add_argument("--zone",  default="LT")
    args = parser.parse_args()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    print(f"git_sha={git_sha}  start={args.start}  end={args.end}  zone={args.zone}")

    df = pull_da_prices(args.start, args.end, zone=args.zone)
    print(f"Pulled {len(df)} hourly records.  NaN count: {df['price_eur_mwh'].isna().sum()}")
    print(df.describe())


if __name__ == "__main__":
    main()
