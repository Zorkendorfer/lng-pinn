"""Pull ENTSO-E day-ahead prices and (optionally) Open-Meteo weather to parquet.

v1.5: optional ``--site`` flag pulls *both* the zone-specific prices and the
location-specific weather in one shot, using the named-site registry in
``lng_pinn.market.SITES``.

Examples:
    # Lithuania (Klaipėda) — preserves the v1.4 default behaviour
    uv run python scripts/01_pull_entsoe.py --start 2021-01-01 --end 2026-01-01

    # Germany (Wilhelmshaven) — pulls DE_LU prices AND Wilhelmshaven weather
    uv run python scripts/01_pull_entsoe.py --start 2021-01-01 --end 2026-01-01 \\
        --site wilhelmshaven

    # Custom site by coordinates
    uv run python scripts/01_pull_entsoe.py --start 2021-01-01 --end 2026-01-01 \\
        --zone DE_LU --lat 53.52 --lon 8.13
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.market import SITES, pull_da_prices, pull_weather, resolve_site


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument(
        "--zone", default=None,
        help="ENTSO-E bidding zone (e.g. LT, DE_LU). Ignored if --site is given.",
    )
    parser.add_argument(
        "--site", default=None,
        help=f"Named FSRU site (overrides --zone). Valid: {', '.join(sorted(SITES))}.",
    )
    parser.add_argument(
        "--lat", type=float, default=None,
        help="Manual override for weather lat (only used if --site is omitted).",
    )
    parser.add_argument(
        "--lon", type=float, default=None,
        help="Manual override for weather lon (only used if --site is omitted).",
    )
    parser.add_argument(
        "--skip-weather", action="store_true",
        help="Skip the Open-Meteo weather pull (useful for re-running after price-only failure).",
    )
    args = parser.parse_args()

    if args.site is not None:
        lat, lon, zone = resolve_site(args.site)
        if args.zone is not None and args.zone != zone:
            print(f"  NOTE: --zone {args.zone} overridden by --site {args.site} → {zone}")
    else:
        zone = args.zone or "LT"
        lat = args.lat
        lon = args.lon

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    print(
        f"git_sha={git_sha}  start={args.start}  end={args.end}  "
        f"zone={zone}  site={args.site or '-'}  lat={lat}  lon={lon}"
    )

    # Prices ------------------------------------------------------------------
    df = pull_da_prices(args.start, args.end, zone=zone)
    print(f"Prices: {len(df)} hourly records.  NaN: {df['price_eur_mwh'].isna().sum()}")
    print(df.describe())

    # Weather (optional, but on by default whenever --site or explicit lat/lon are given,
    # or for the legacy Klaipėda default) ------------------------------------
    if args.skip_weather:
        return
    weather_kwargs: dict[str, float] = {}
    if lat is not None:
        weather_kwargs["lat"] = lat
    if lon is not None:
        weather_kwargs["lon"] = lon
    print()
    print(f"Weather: pulling Open-Meteo @ {weather_kwargs or '(default Klaipėda)'}")
    wdf = pull_weather(args.start, args.end, **weather_kwargs)
    print(f"Weather: {len(wdf)} hourly records.  NaN: {wdf.isna().sum().sum()}")


if __name__ == "__main__":
    main()
