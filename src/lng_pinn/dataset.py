"""Build training dataset and hourly timeseries from the CoolProp plant simulator."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

from lng_pinn.composition import build_composition_series
from lng_pinn.market import load_da_prices, pull_weather

PROCESSED_DIR = Path("data/processed")

# Operating envelope bounds  [min, max]
BOUNDS = {
    "CH4": (0.82, 0.96),
    "C2H6": (0.02, 0.12),
    "C3H8": (0.005, 0.035),
    "nC4H10": (0.001, 0.015),
    "iC4H10": (0.001, 0.010),
    # N2 is computed as remainder
    "m_dot": (10.0, 80.0),  # kg/s
    "T_amb": (258.0, 308.0),  # K
    "T_sw": (271.0, 298.0),  # K
}

# Stratified sampling: concentrate extra points at low flow where PINN error is highest.
# 30% of samples drawn from m_dot ∈ [M_DOT_MIN, LOW_MDOT_MAX]; rest from full range.
LOW_MDOT_FRAC = 0.30
LOW_MDOT_MAX = 25.0  # kg/s


def _sample_compositions(lhs_cols: np.ndarray) -> np.ndarray:
    """Map first 5 LHS columns to valid mole fractions (sum-to-1 constrained)."""
    raw = lhs_cols.copy()
    ranges = [(0.82, 0.96), (0.02, 0.12), (0.005, 0.035), (0.001, 0.015), (0.001, 0.010)]
    for i, (lo, hi) in enumerate(ranges):
        raw[:, i] = lo + raw[:, i] * (hi - lo)
    n2 = np.clip(1.0 - raw.sum(axis=1), 0.0, 0.02)
    total = raw.sum(axis=1) + n2
    raw = raw / total[:, None]
    n2 = n2 / total
    return np.column_stack([raw, n2])


def _simulate_one(args: tuple[Any, ...]) -> dict[str, float] | None:
    """Top-level function so it can be pickled by ProcessPoolExecutor."""
    import CoolProp.CoolProp as CP

    from lng_pinn.plant import (  # imported inside worker to avoid pickling issues
        J_TO_KWH,
        P_IN,
        P_OUT_DEFAULT,
        T_SENDOUT,
        pump_efficiency,
        simulate,
    )
    from lng_pinn.thermo import get_state

    x, m_dot, T_amb, T_sw = args
    try:
        out = simulate(x, float(m_dot), float(T_amb), float(T_sw))
    except ValueError:
        # CoolProp can't find a density solution (two-phase or near-critical region)
        return None

    # Enthalpy at storage and send-out conditions (needed for PINN energy balance loss).
    # Storage is saturated liquid at P_IN (PQ_INPUTS with Q=0); doing PT_INPUTS at a
    # fixed T_IN=111 K returns gas-phase density for high-N2 mixtures whose bubble
    # point sits below 111 K.
    try:
        state = get_state(x)
        state.update(CP.PQ_INPUTS, P_IN, 0.0)
        h_in_per_kg = state.hmolar() / state.molar_mass()    # J/kg
        rho_in = state.rhomass()                             # kg/m^3
        state.update(CP.PT_INPUTS, P_OUT_DEFAULT, T_SENDOUT)
        h_out_per_kg = state.hmolar() / state.molar_mass()   # J/kg
    except ValueError:
        return None  # composition not liquid at storage conditions

    # Analytical pump work (incompressible liquid, flow-dependent efficiency)
    eta = pump_efficiency(float(m_dot))
    W_pump_expected = (P_OUT_DEFAULT - P_IN) / rho_in / eta * J_TO_KWH  # kWh/kg

    return {
        "CH4": x[0],
        "C2H6": x[1],
        "C3H8": x[2],
        "nC4H10": x[3],
        "iC4H10": x[4],
        "N2": x[5],
        "m_dot": float(m_dot),
        "T_amb": float(T_amb),
        "T_sw": float(T_sw),
        "W_pump": out.W_pump,
        "W_trim": out.W_trim,
        "W_total": out.W_total,
        "T_out": out.T_out,
        "Q_sw": out.Q_sw,
        "exergy_destruction": out.exergy_destruction,
        "h_in_per_kg": h_in_per_kg,
        "h_out_per_kg": h_out_per_kg,
        "W_pump_expected": W_pump_expected,
    }


def _generate_sample_args(N: int, seed: int) -> list[tuple[Any, ...]]:
    """Deterministically generate the N stratified-LHS operating-point arguments.

    Given the same (N, seed), this returns identical arguments — that determinism is
    what lets `build_training_set` resume safely from a partial checkpoint.
    """
    m_lo, m_hi = BOUNDS["m_dot"]
    N_low = int(N * LOW_MDOT_FRAC)
    N_main = N - N_low

    s_main = LatinHypercube(d=8, seed=seed).random(N_main)
    comp_main = _sample_compositions(s_main[:, :5])
    m_dot_main = m_lo + s_main[:, 5] * (m_hi - m_lo)
    T_amb_main = BOUNDS["T_amb"][0] + s_main[:, 6] * (BOUNDS["T_amb"][1] - BOUNDS["T_amb"][0])
    T_sw_main = BOUNDS["T_sw"][0] + s_main[:, 7] * (BOUNDS["T_sw"][1] - BOUNDS["T_sw"][0])

    s_low = LatinHypercube(d=8, seed=seed + 1).random(N_low)
    comp_low = _sample_compositions(s_low[:, :5])
    m_dot_low = m_lo + s_low[:, 5] * (LOW_MDOT_MAX - m_lo)
    T_amb_low = BOUNDS["T_amb"][0] + s_low[:, 6] * (BOUNDS["T_amb"][1] - BOUNDS["T_amb"][0])
    T_sw_low = BOUNDS["T_sw"][0] + s_low[:, 7] * (BOUNDS["T_sw"][1] - BOUNDS["T_sw"][0])

    compositions = np.vstack([comp_main, comp_low])
    m_dot = np.concatenate([m_dot_main, m_dot_low])
    T_amb = np.concatenate([T_amb_main, T_amb_low])
    T_sw = np.concatenate([T_sw_main, T_sw_low])

    return [(tuple(compositions[i]), m_dot[i], T_amb[i], T_sw[i]) for i in range(N)]


def _write_partial(records: dict[int, dict[str, float]], path: Path) -> None:
    """Atomic write of the per-id records dict to a parquet checkpoint."""
    if not records:
        return
    df = pd.DataFrame([records[i] for i in sorted(records.keys())])
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def build_training_set(
    N: int = 20_000,
    seed: int = 0,
    workers: int | None = None,
    resume: bool = True,
    ckpt_every: int = 2000,
) -> pd.DataFrame:
    """Sample N operating points via stratified LHS, simulate in parallel, return DataFrame.

    Columns: CH4, C2H6, C3H8, nC4H10, iC4H10, N2, m_dot, T_amb, T_sw,
             W_pump, W_trim, W_total, T_out, Q_sw, exergy_destruction,
             h_in_per_kg, h_out_per_kg, W_pump_expected

    Sampling strategy: LOW_MDOT_FRAC of points are drawn from m_dot ∈ [M_DOT_MIN,
    LOW_MDOT_MAX] to reduce PINN error in the low-flow regime used heavily by dispatch.

    Resume behaviour: a per-(N, seed) partial parquet is flushed every `ckpt_every`
    completed samples. If the process is killed and rerun with the same (N, seed),
    the partial is loaded and only the missing sample IDs are submitted to the pool.
    The partial file is removed once the final train.parquet is written.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = PROCESSED_DIR / f"train_partial_N{N}_s{seed}.parquet"

    args = _generate_sample_args(N, seed)

    # Resume: load any prior records (keyed by deterministic sample id 0..N-1).
    done: dict[int, dict[str, float]] = {}
    if resume and partial_path.exists():
        try:
            prior = pd.read_parquet(partial_path)
            if "_sample_id" in prior.columns:
                for rec in prior.to_dict(orient="records"):
                    sid = int(rec.pop("_sample_id"))
                    done[sid] = rec
                print(f"  Resuming from {partial_path.name}: {len(done)}/{N} samples cached")
            else:
                print(f"  Ignoring {partial_path.name}: missing _sample_id column")
        except Exception as exc:
            print(f"  Could not resume from {partial_path.name}: {exc}")

    pending = [(i, args[i]) for i in range(N) if i not in done]

    if pending:
        n_workers = workers or max(1, (os.cpu_count() or 1))
        completed_since_ckpt = 0

        def _record_result(sid: int, rec: dict[str, float] | None) -> None:
            nonlocal completed_since_ckpt
            if rec is not None:
                rec["_sample_id"] = sid
                done[sid] = rec
            completed_since_ckpt += 1
            if completed_since_ckpt >= ckpt_every:
                _write_partial(done, partial_path)
                completed_since_ckpt = 0

        desc = f"Simulating ({len(pending)} new)"
        if n_workers == 1:
            for sid, arg in tqdm(pending, total=len(pending), desc=desc, unit="pts"):
                _record_result(sid, _simulate_one(arg))
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_id = {executor.submit(_simulate_one, a): sid for sid, a in pending}
                for future in tqdm(
                    as_completed(future_to_id), total=len(pending), desc=desc, unit="pts"
                ):
                    sid = future_to_id[future]
                    _record_result(sid, future.result())

        # Final flush so the partial reflects everything we just computed.
        _write_partial(done, partial_path)

    records = [done[i] for i in sorted(done.keys())]
    n_skipped = N - len(records)
    if n_skipped:
        import warnings
        warnings.warn(
            f"Skipped {n_skipped}/{N} ({100*n_skipped/N:.1f}%) points where CoolProp "
            "found no density solution (two-phase or near-critical region).",
            stacklevel=2,
        )

    df = pd.DataFrame(records)
    if "_sample_id" in df.columns:
        df = df.drop(columns=["_sample_id"])
    df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)

    # Clean up partial once final is on disk.
    if partial_path.exists():
        partial_path.unlink()

    return df


def append_trajectory_rows(
    rows: list[dict],
    dedupe_tol: float = 1e-4,
) -> int:
    """v1.3 A2: append trajectory-labelled rows to data/processed/train.parquet.

    ``rows`` is the list of dicts produced by ``_simulate_one`` for points
    sampled on the dispatch trajectory. Each row is tagged with
    ``_source="trajectory"``. Rows whose (composition, m_dot, T_amb, T_sw)
    are within ``dedupe_tol`` of an existing row in the training set are
    skipped — LHS already covers most of the input space, so we only want
    to add genuinely new points.

    Returns the number of rows actually appended after deduplication.
    """
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} does not exist; run scripts/02_build_dataset.py first."
        )
    existing = pd.read_parquet(train_path)
    if "_source" not in existing.columns:
        existing["_source"] = "lhs"

    new_df = pd.DataFrame(rows)
    new_df["_source"] = "trajectory"

    key_cols = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "N2", "m_dot", "T_amb", "T_sw"]
    # Round to a tolerance bucket so near-duplicates collapse.
    decimals = max(0, int(round(-np.log10(dedupe_tol))))
    existing_arr = existing[key_cols].values
    existing_keys = {
        tuple(np.round(existing_arr[i], decimals)) for i in range(len(existing_arr))
    }
    new_arr = new_df[key_cols].values
    keep_mask = np.array([
        tuple(np.round(new_arr[i], decimals)) not in existing_keys
        for i in range(len(new_arr))
    ])
    new_df = new_df.loc[keep_mask].reset_index(drop=True)
    if len(new_df) == 0:
        return 0

    combined = pd.concat([existing, new_df], ignore_index=True)
    tmp = train_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(train_path)
    return int(len(new_df))


def timeseries_path(zone: str = "LT") -> Path:
    """Canonical timeseries parquet path for a price zone.

    LT keeps the historical bare name ``timeseries.parquet`` for backward
    compatibility; other zones (v1.4 B) are suffixed, e.g.
    ``timeseries_DE.parquet``, so a second-zone run never clobbers the first.
    """
    if zone == "LT":
        return PROCESSED_DIR / "timeseries.parquet"
    return PROCESSED_DIR / f"timeseries_{zone}.parquet"


def build_timeseries(
    start: str = "2021-01-01",
    end: str = "2026-01-01",
    zone: str = "LT",
    seed: int = 42,
    site: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> pd.DataFrame:
    """Build hourly timeseries of (price, T_amb, T_sw, composition) for dispatch.

    Joins ENTSO-E day-ahead prices, Open-Meteo weather, and synthetic
    cargo composition trajectories on a common UTC hourly index.

    v1.5: site-aware. Pass either a named ``site`` (resolves to lat/lon/zone
    via :data:`lng_pinn.market.SITES`) or explicit ``lat`` and ``lon``. If
    only ``site`` is given, ``zone`` is overridden to match the site's
    bidding zone. If neither is given, falls back to the Klaipėda defaults
    so all v1.4 call sites keep working unchanged.

    Returns:
        DataFrame saved to data/processed/timeseries[_<zone>].parquet with columns:
        price_eur_mwh, T_amb (K), T_sw (K), CH4, C2H6, C3H8, nC4H10, iC4H10, N2.
    """
    from lng_pinn.market import LAT as DEFAULT_LAT
    from lng_pinn.market import LON as DEFAULT_LON
    from lng_pinn.market import resolve_site

    if site is not None:
        s_lat, s_lon, s_zone = resolve_site(site)
        if lat is None:
            lat = s_lat
        if lon is None:
            lon = s_lon
        # A named site overrides the zone unless an explicit non-default
        # zone was passed in.
        if zone == "LT":
            zone = s_zone
    if lat is None:
        lat = DEFAULT_LAT
    if lon is None:
        lon = DEFAULT_LON

    prices = load_da_prices(start, end, zone=zone)
    weather = pull_weather(start, end, lat=lat, lon=lon)

    # Align weather to price index (resample/reindex to hourly UTC)
    idx = prices.index
    weather = weather.reindex(idx, method="nearest", tolerance="1h")

    # Synthetic composition on same index
    comp = build_composition_series(idx, seed=seed)

    ts = pd.concat([prices, weather, comp], axis=1)
    ts = ts.dropna(subset=["price_eur_mwh"])  # drop any residual gaps

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(timeseries_path(zone))
    return ts
