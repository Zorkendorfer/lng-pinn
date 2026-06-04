"""Benchmark PINN surrogate inference against CoolProp simulation.

The benchmark samples rows from data/processed/train.parquet, evaluates the
trained surrogate in one batch, evaluates the same rows through the CoolProp
plant simulator serially, and writes results/tables/speed_benchmark.csv.

This is intentionally separate from the manuscript-number checker because it is
machine-dependent and should be reported with hardware context if used.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.pinn import build_aux, load
from lng_pinn.plant import simulate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
COMP_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2"]
INPUT_COLS = COMP_COLS + ["m_dot", "T_amb", "T_sw"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of rows to benchmark")
    parser.add_argument("--seed", type=int, default=123, help="sample seed")
    parser.add_argument(
        "--train",
        default=str(ROOT / "data" / "processed" / "train.parquet"),
        help="training parquet to sample benchmark points from",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.train)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    model, scaler = load()
    # Parquet-backed pandas arrays can expose read-only NumPy views; PyTorch
    # warns because tensors created from those views would be unsafe to mutate.
    X_np = sample[INPUT_COLS].to_numpy(dtype="float32", copy=True)
    aux = build_aux(X_np[:, :6], X_np[:, 6])
    X = torch.from_numpy(X_np)

    # Warm up the torch path once so first-call overhead is not counted.
    with torch.no_grad():
        _ = scaler.unscale_y(model(scaler.scale_x(X[: min(16, len(X))]), aux[: min(16, len(X))]))

    t0 = time.perf_counter()
    for _ in tqdm(range(1), desc="PINN batch", unit="batch"):
        with torch.no_grad():
            _ = scaler.unscale_y(model(scaler.scale_x(X), aux, scaler=scaler)).numpy()
    pinn_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    failures = 0
    for row in tqdm(
        sample.itertuples(index=False),
        total=len(sample),
        desc="CoolProp",
        unit="pt",
    ):
        comp = tuple(float(getattr(row, c)) for c in COMP_COLS)
        try:
            simulate(comp, float(row.m_dot), float(row.T_amb), float(row.T_sw))
        except ValueError:
            failures += 1
    coolprop_s = time.perf_counter() - t0

    n_ok = len(sample) - failures
    speedup = (coolprop_s / pinn_s) if pinn_s > 0 else float("nan")
    out = pd.DataFrame([{
        "n_requested": int(args.n),
        "n_evaluated": int(len(sample)),
        "coolprop_failures": int(failures),
        "pinn_seconds": float(pinn_s),
        "coolprop_seconds": float(coolprop_s),
        "pinn_ms_per_point": float(pinn_s / len(sample) * 1000.0),
        "coolprop_ms_per_point": float(coolprop_s / max(n_ok, 1) * 1000.0),
        "speedup_x": float(speedup),
        "python": platform.python_version(),
        "machine": platform.platform(),
        "processor": platform.processor(),
    }])

    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "speed_benchmark.csv", index=False)
    print(out.to_string(index=False))
    print("wrote results/tables/speed_benchmark.csv")


if __name__ == "__main__":
    main()
