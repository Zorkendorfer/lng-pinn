"""Write a reproducibility manifest for the code-side rework runs."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("results/tables")
MODELS_DIR = Path("results/models")
DEFAULT_EXPECTED = [
    "results/models/pinn_v1.pt",
    "results/models/pinn_soft.pt",
    "results/tables/mixing_table3.csv",
    "results/tables/soft_vs_hard.csv",
    "results/tables/soft_vs_hard_contrast.csv",
    "results/tables/carbon_ensemble.csv",
    "results/tables/fabrication_diagnostic.csv",
    "results/tables/phase2_validation_composition.csv",
]


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _file_entry(path: str) -> dict[str, object]:
    p = Path(path)
    if not p.exists():
        return {"path": path, "exists": False}
    stat = p.stat()
    return {
        "path": path,
        "exists": True,
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _commands() -> list[str]:
    prices = "0 20 40 60 80 100 120 160"
    return [
        "uv run python scripts/03_train_pinn.py --arch soft --lambda-e 1.0 --lambda-p 1.0 --out results/models/pinn_soft.pt --fresh",
        "uv run python scripts/09_mixing_sensitivity.py --carbon-price 80 --workers 10",
        "uv run python scripts/14_mixing_table.py --strict",
        "uv run python scripts/06_seed_sensitivity.py --carbon-price 0 --surrogate hard --workers 6",
        "uv run python scripts/06_seed_sensitivity.py --carbon-price 80 --surrogate hard --workers 6",
        "uv run python scripts/06_seed_sensitivity.py --carbon-price 0 --surrogate soft --model-path results/models/pinn_soft.pt --workers 6",
        "uv run python scripts/06_seed_sensitivity.py --carbon-price 80 --surrogate soft --model-path results/models/pinn_soft.pt --workers 6",
        "uv run python scripts/12_soft_vs_hard.py --strict",
        "uv run python scripts/13_carbon_ensemble.py --prices " + prices + " --strict",
        "uv run python scripts/11_fabrication_diagnostic.py --surrogate hard",
        "uv run python scripts/11_fabrication_diagnostic.py --surrogate soft --model-path results/models/pinn_soft.pt",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR / "run_manifest.json"))
    parser.add_argument("--expected", nargs="+", default=DEFAULT_EXPECTED)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": _cmd(["git", "branch", "--show-current"]),
        "git_status_short": _cmd(["git", "status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "commands": _commands(),
        "artifacts": [_file_entry(path) for path in args.expected],
    }
    missing = [a["path"] for a in manifest["artifacts"] if not a["exists"]]
    manifest["missing_artifacts"] = missing

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    if missing:
        print("missing artifacts:")
        for path in missing:
            print(f"  {path}")
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
