"""Sanity-check paper-facing artifacts for stale or inconsistent outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
PAPER = ROOT / "paper"
PROCESSED = ROOT / "data" / "processed"

EXPECTED_FIGURES = [
    "fig_carbon_ensemble.pdf",
    "fig6_carbon_sweep.pdf",
    "fig6_carbon_sweep_2021.pdf",
    "fig6_carbon_sweep_2022.pdf",
    "fig6_carbon_sweep_2023.pdf",
    "fig6_carbon_sweep_2024.pdf",
    "fig6_carbon_sweep_2025.pdf",
    "fig8_volatility_vs_saving.pdf",
]

EXPECTED_TABLES = [
    "seed_significance_hard_co280.csv",
    "carbon_ensemble.csv",
    "cost_decomposition.csv",
    "cost_decomposition_delta.csv",
    "cost_decomposition_summary.csv",
    "paper_numbers.json",
    "mixing_table3.csv",
    "soft_vs_hard.csv",
    "soft_vs_hard_contrast.csv",
    "speed_benchmark.csv",
    "surrogate_eval.csv",
    "tout_audit.csv",
]


def _require(path: Path, errors: list[str]) -> bool:
    if not path.exists():
        errors.append(f"missing {path.relative_to(ROOT)}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        errors.append(f"empty {path.relative_to(ROOT)}")
        return False
    return True


def _audit_seed_significance(errors: list[str]) -> None:
    path = RESULTS / "seed_significance_hard_co280.csv"
    if not _require(path, errors):
        return
    df = pd.read_csv(path)
    expected = {str(y) for y in range(2021, 2026)} | {"ALL_5yr_mean"}
    scopes = set(df[df["baseline"].astype(str) == "lagged"]["scope"].astype(str))
    missing = expected - scopes
    if missing:
        errors.append(f"{path.name}: missing scopes {sorted(missing)}")
    if "wilcoxon_p_two_sided" not in df.columns:
        errors.append(f"{path.name}: missing wilcoxon_p_two_sided")


def _audit_mixing(errors: list[str]) -> None:
    path = RESULTS / "mixing_table3.csv"
    if not _require(path, errors):
        return
    df = pd.read_csv(path)
    expected_taus = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}
    present = {float(x) for x in df["tau_days"].dropna()}
    if expected_taus - present:
        errors.append(f"{path.name}: missing tau values {sorted(expected_taus - present)}")
    for col in ["linear_n", "exp_n"]:
        if col not in df.columns:
            errors.append(f"{path.name}: missing {col}")
        elif (df[col].astype(int) < 10).any():
            errors.append(f"{path.name}: expected seed-first n >= 10 in {col}")


def _audit_volatility(errors: list[str]) -> None:
    path = RESULTS / "volatility_vs_saving.csv"
    if not _require(path, errors):
        return
    df = pd.read_csv(path)
    if "year" not in df.columns:
        errors.append(f"{path.name}: missing year column")
        return
    years = {int(y) for y in df["year"].dropna().unique()}
    expected = set(range(2021, 2026))
    if years != expected:
        errors.append(f"{path.name}: expected years {sorted(expected)}, got {sorted(years)}")


def _audit_paper_numbers(errors: list[str]) -> None:
    json_path = RESULTS / "paper_numbers.json"
    macros_path = PAPER / "paper_macros.tex"
    if not json_path.exists() or not macros_path.exists():
        return
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{json_path.name}: malformed JSON ({exc})")
        return
    for key in ["headline_saving_vs_lagged_pct", "record_count_market_hours"]:
        if key not in data:
            errors.append(f"{json_path.name}: missing {key}")


def _audit_timeseries(errors: list[str]) -> None:
    path = PROCESSED / "timeseries.parquet"
    if not _require(path, errors):
        return
    ts = pd.read_parquet(path)
    idx = pd.DatetimeIndex(pd.to_datetime(ts.index, utc=True))
    if not idx.is_unique:
        errors.append(f"{path.name}: index contains duplicate timestamps")
    if not idx.is_monotonic_increasing:
        errors.append(f"{path.name}: index is not monotonic increasing")

    expected_idx = pd.date_range(
        "2021-01-01",
        "2026-01-01",
        freq="h",
        inclusive="left",
        tz="UTC",
    )
    if len(idx) != len(expected_idx):
        errors.append(f"{path.name}: expected {len(expected_idx)} hourly rows, got {len(idx)}")
    if len(idx) and (idx[0] != expected_idx[0] or idx[-1] != expected_idx[-1]):
        errors.append(
            f"{path.name}: expected UTC span {expected_idx[0]}..{expected_idx[-1]}, "
            f"got {idx[0]}..{idx[-1]}"
        )
    if len(idx) == len(expected_idx) and not idx.equals(expected_idx):
        errors.append(f"{path.name}: timestamps are not the exact 2021-2025 hourly grid")

    counts = pd.Series(1, index=idx).groupby(idx.year).sum().to_dict()
    expected_counts = {2021: 8760, 2022: 8760, 2023: 8760, 2024: 8784, 2025: 8760}
    if counts != expected_counts:
        errors.append(f"{path.name}: expected year counts {expected_counts}, got {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--timeseries-only",
        action="store_true",
        help="Only check that data/processed/timeseries.parquet is a uniform hourly grid.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if args.timeseries_only:
        _audit_timeseries(errors)
        if errors:
            print("Timeseries audit found issues:")
            for err in errors:
                print(f"  - {err}")
            if args.strict:
                raise SystemExit(2)
        else:
            print("Timeseries audit passed.")
        return

    for name in EXPECTED_FIGURES:
        _require(FIGURES / name, errors)
    for name in EXPECTED_TABLES:
        _require(RESULTS / name, errors)
    _require(PAPER / "paper_macros.tex", errors)
    _audit_timeseries(errors)
    _audit_seed_significance(errors)
    _audit_mixing(errors)
    _audit_volatility(errors)
    _audit_paper_numbers(errors)

    if errors:
        print("Artifact audit found issues:")
        for err in errors:
            print(f"  - {err}")
        if args.strict:
            raise SystemExit(2)
    else:
        print("Artifact audit passed.")


if __name__ == "__main__":
    main()
