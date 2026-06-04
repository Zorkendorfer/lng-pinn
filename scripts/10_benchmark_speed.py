#!/usr/bin/env python3
"""Benchmark CoolProp HEOS plant calls against surrogate forward passes.

The manuscript reports the median speedup from 10,000 full plant simulations
versus 10,000 neural-surrogate forward passes on the same sampled operating
points. Surrogate timing uses precomputed auxiliary thermodynamic columns,
matching the intended "forward pass" comparison rather than re-timing CoolProp
inside the auxiliary lookup.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lng_pinn.pinn import load
from lng_pinn.plant import simulate

INPUT_COLS = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2", "m_dot", "T_amb", "T_sw"]
AUX_COLS = ["h_in_per_kg", "h_out_per_kg", "W_pump_expected"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--surrogate-repeats", type=int, default=9)
    ap.add_argument("--coolprop-repeats", type=int, default=3)
    args = ap.parse_args()

    df = (
        pd.read_parquet("data/processed/train.parquet", columns=INPUT_COLS + AUX_COLS)
        .sample(n=args.n, random_state=args.seed)
        .reset_index(drop=True)
    )

    torch.set_num_threads(1)
    model, scaler = load()
    model.eval()
    x = torch.tensor(df[INPUT_COLS].to_numpy("float32"))
    aux = torch.tensor(df[AUX_COLS].to_numpy("float32"))
    x_norm = scaler.scale_x(x)

    with torch.inference_mode():
        model(x_norm, aux)

    surrogate_times = []
    for _ in tqdm(range(args.surrogate_repeats), desc="Surrogate repeats", unit="rep"):
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(x_norm, aux)
        surrogate_times.append(time.perf_counter() - t0)

    records = list(df[INPUT_COLS].itertuples(index=False, name=None))
    coolprop_times = []
    for _ in tqdm(range(args.coolprop_repeats), desc="CoolProp repeats", unit="rep"):
        t0 = time.perf_counter()
        for row in tqdm(
            records,
            desc="  CoolProp calls",
            unit="pt",
            leave=False,
        ):
            simulate(
                tuple(float(v) for v in row[:6]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
            )
        coolprop_times.append(time.perf_counter() - t0)

    surrogate_median = statistics.median(surrogate_times)
    coolprop_median = statistics.median(coolprop_times)
    speedup = coolprop_median / surrogate_median

    print(f"n={args.n}")
    print(f"surrogate_median_s={surrogate_median:.6f}")
    print(f"coolprop_median_s={coolprop_median:.6f}")
    print(f"speedup={speedup:.1f}")
    print(f"surrogate_calls_per_s={args.n / surrogate_median:.1f}")
    print(f"coolprop_calls_per_s={args.n / coolprop_median:.1f}")

    out = pd.DataFrame([{
        "n": int(args.n),
        "surrogate_repeats": int(args.surrogate_repeats),
        "coolprop_repeats": int(args.coolprop_repeats),
        "surrogate_median_s": float(surrogate_median),
        "coolprop_median_s": float(coolprop_median),
        "speedup_x": float(speedup),
        "surrogate_calls_per_s": float(args.n / surrogate_median),
        "coolprop_calls_per_s": float(args.n / coolprop_median),
        "python": platform.python_version(),
        "machine": platform.platform(),
        "processor": platform.processor(),
    }])
    out_path = Path("results/tables/speed_benchmark_repeated.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
