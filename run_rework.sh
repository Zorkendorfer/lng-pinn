#!/usr/bin/env bash
# run_rework.sh — code_rework_plan.md execution runbook
#
# Six phases, each independently runnable. Total wall-clock on a 12-core
# M-series with --workers 10 and v1.5 E2 validation-sample mode:
#   Phase 0 (tests):          ~10 s
#   Phase 1 (storage-phase):  ~3 s
#   Phase 2 (CSTR sweep):     ~2–4 hours
#   Phase 3 (soft null check):~3 hours
#   Phase 4 (fabrication/null diagnostic): ~30 min
#   Phase 5 (carbon ensemble):~3–4 hours
#   Phase 6 (manifest+figs+audits): ~30 min
#   ----------------------------------
#   Full:                     ~10–12 hours
#
# Pass a phase name (or "all") as the first arg:
#   ./run_rework.sh tests
#   ./run_rework.sh cstr
#   ./run_rework.sh soft
#   ./run_rework.sh fabrication
#   ./run_rework.sh ensemble
#   ./run_rework.sh manifest
#   ./run_rework.sh all
#
# Each phase is idempotent — caches in data/processed/ and results/tables/
# let interrupted runs resume cleanly.

set -e
PHASE="${1:-help}"
CARBON_PRICE="${CARBON_PRICE:-80}"
WORKERS="${WORKERS:-10}"
VAL_FRAC="${VAL_FRAC:-0.05}"      # v1.5 E2 validation-sample fraction
PRICES_AXIS="${PRICES_AXIS:-0 20 40 60 80 100 120 160}"
FAB_WINDOWS="${FAB_WINDOWS:-12}"  # CoolProp-heavy windows per fabrication diagnostic
MIXING_RESUME="${MIXING_RESUME:-0}"  # 0 after hourly data fixes; 1 only to resume same data

echo "=== run_rework.sh phase=$PHASE  carbon_price=$CARBON_PRICE  workers=$WORKERS  fab_windows=$FAB_WINDOWS ==="

require_hourly_timeseries() {
  echo "[data] verifying uniform hourly timeseries"
  uv run python scripts/18_audit_artifacts.py --timeseries-only --strict
}

# ---------------------------------------------------------------------------
phase_tests() {
  echo "[tests] running full pytest suite — guards storage-phase regression + new kernels"
  uv run pytest tests/ -q
}

# ---------------------------------------------------------------------------
# PHASE 1 — item 5 (storage-phase regression test)
phase_storage_phase() {
  echo "[storage_phase] item 5 — assert liquid density across composition envelope"
  uv run pytest tests/test_storage_phase.py -v
}

# ---------------------------------------------------------------------------
# PHASE 2 — item 1 (CSTR mixing-kernel sweep + Table 3)
# Grid: tau_mix ∈ {1,2,3,5,7,10} d × kernels ∈ {linear, exp} × 10 seeds × 5 yrs.
# 09 has its own cache; the run is resumable if interrupted.
phase_cstr() {
  require_hourly_timeseries
  echo "[cstr] item 1 — full tau × kernel × seed × year sweep"
  RESUME_FLAG="--no-resume"
  if [ "$MIXING_RESUME" = "1" ]; then
    RESUME_FLAG=""
  fi
  uv run python scripts/09_mixing_sensitivity.py \
      --carbon-price "$CARBON_PRICE" \
      $RESUME_FLAG \
      --workers "$WORKERS"
  echo "[cstr] formatting Table 3 from sensitivity output"
  uv run python scripts/14_mixing_table.py --strict
}

# ---------------------------------------------------------------------------
# PHASE 3 — item 2 (soft-vs-hard controlled null check)
# Trains a soft surrogate then runs four tagged seed-sensitivity backtests:
#   {hard, soft} × {co2=0, co2=80}, all 10 seeds.
phase_soft() {
  require_hourly_timeseries
  echo "[soft] item 2a — train soft surrogate (~30 min)"
  if [ ! -f results/models/pinn_soft.pt ]; then
    uv run python scripts/03_train_pinn.py --no-resume --arch soft
  else
    echo "  results/models/pinn_soft.pt already exists, skipping training"
  fi

  for ARCH in hard soft; do
    MODEL_FLAG="results/models/pinn_v1.pt"
    [ "$ARCH" = "soft" ] && MODEL_FLAG="results/models/pinn_soft.pt"
    for CP in 0 "$CARBON_PRICE"; do
      echo "[soft] item 2b — 10-seed backtest: arch=$ARCH carbon=$CP"
      uv run python scripts/06_seed_sensitivity.py \
          --no-resume \
          --model-path "$MODEL_FLAG" \
          --surrogate "$ARCH" \
          --carbon-price "$CP" \
          --workers "$WORKERS" \
          --validation-sample-frac "$VAL_FRAC"
    done
  done

  echo "[soft] item 2c — aggregate into soft_vs_hard.csv"
  uv run python scripts/12_soft_vs_hard.py
}

# ---------------------------------------------------------------------------
# PHASE 4 — item 3 (fabrication diagnostic, executable null check)
phase_fabrication() {
  require_hourly_timeseries
  for ARCH in hard soft; do
    MODEL_FLAG="results/models/pinn_v1.pt"
    [ "$ARCH" = "soft" ] && MODEL_FLAG="results/models/pinn_soft.pt"
    if [ ! -f "$MODEL_FLAG" ]; then
      echo "[fabrication] skipping arch=$ARCH — $MODEL_FLAG missing (run phase 'soft' first)"
      continue
    fi
    for SEED in 42 0 1 7 13; do
      echo "[fabrication] item 3 — diagnostic: arch=$ARCH seed=$SEED"
      uv run python scripts/11_fabrication_diagnostic.py \
          --surrogate "$ARCH" \
          --model-path "$MODEL_FLAG" \
          --seed "$SEED" \
          --carbon-price "$CARBON_PRICE" \
          --max-windows "$FAB_WINDOWS"
    done
  done
}

# ---------------------------------------------------------------------------
# PHASE 5 — item 4 (10-seed × full carbon axis ensemble)
# Runs the seed sensitivity at every carbon price in PRICES_AXIS, then
# aggregates into carbon_ensemble.csv (seed-mean ± CI across the whole axis).
phase_ensemble() {
  require_hourly_timeseries
  for CP in $PRICES_AXIS; do
    echo "[ensemble] item 4 — 10-seed backtest at carbon=$CP"
    uv run python scripts/06_seed_sensitivity.py \
        --no-resume \
        --surrogate hard \
        --carbon-price "$CP" \
        --workers "$WORKERS" \
        --validation-sample-frac "$VAL_FRAC"
  done
  echo "[ensemble] item 4 — aggregate carbon_ensemble.csv"
  uv run python scripts/13_carbon_ensemble.py
}

# ---------------------------------------------------------------------------
# PHASE V1 — fix_strategy.md 2026-06-12 TASK V1 (volume-matched comparison)
# Pins realised delivered volume via rolling volume-debt accounting plus a
# narrow demand band, so the aware-vs-lagged comparison has no volume channel.
# Artifacts are tagged hard_volmatch_co2<P> and never touch the canonical
# floor-constraint results. ~30 min on 10 workers.
phase_volmatch() {
  require_hourly_timeseries
  echo "[volmatch] TASK V1 — 10-seed volume-matched backtest at carbon=$CARBON_PRICE"
  uv run python scripts/06_seed_sensitivity.py \
      --no-resume \
      --surrogate hard \
      --volume-matched \
      --carbon-price "$CARBON_PRICE" \
      --workers "$WORKERS" \
      --validation-sample-frac "$VAL_FRAC"

  echo "[volmatch] TASK V1 — decomposition replay (acceptance: |mass delta| <= 0.1%)"
  uv run python scripts/16_cost_decomposition.py \
      --surrogate hard_volmatch \
      --carbon-price "$CARBON_PRICE" \
      --out results/tables/cost_decomposition_volmatch.csv \
      --delta-out results/tables/cost_decomposition_delta_volmatch.csv \
      --summary-out results/tables/cost_decomposition_summary_volmatch.csv \
      --strict

  echo "[volmatch] refreshing paper macros + generated tables"
  uv run python scripts/09_paper_numbers.py --write
  echo "[volmatch] done — rebuild the manuscript with: (cd paper && latexmk -pdf main.tex)"
}

# ---------------------------------------------------------------------------
# PHASE 6 — item 10 (manifest + figure refresh)
phase_manifest() {
  require_hourly_timeseries
  echo "[manifest] item 10 — write run_manifest.json"
  uv run python scripts/15_run_manifest.py

  echo "[manifest] regenerating figures from refreshed tables"
  uv run python scripts/05_make_figures.py --no-resume --carbon-price "$CARBON_PRICE"
  uv run python -c "
import pandas as pd
from pathlib import Path
from lng_pinn.plots import fig_carbon_sweep, fig_carbon_sweep_per_year
parts = [pd.read_csv(p).assign(price_co2_eur_per_t=float(p.stem.replace('carbon_sweep_co2_','')))
         for p in sorted(Path('results/tables').glob('carbon_sweep_co2_*.csv'))]
if parts:
    sweep = pd.concat(parts, ignore_index=True).sort_values(['price_co2_eur_per_t','year']).reset_index(drop=True)
    sweep.to_csv('results/tables/carbon_sweep.csv', index=False)
    fig_carbon_sweep(sweep); fig_carbon_sweep_per_year(sweep)
    print('fig6 + per-year companions re-rendered')
"
  echo "[manifest] cheap post-processing audits"
  uv run python scripts/16_cost_decomposition.py \
      --surrogate hard \
      --carbon-price "$CARBON_PRICE" \
      --strict
  uv run python scripts/17_tout_audit.py
  uv run python scripts/09_paper_numbers.py --write --refresh-validation
  uv run python scripts/18_audit_artifacts.py --strict
}

# ---------------------------------------------------------------------------
case "$PHASE" in
  tests)       phase_tests ;;
  storage_phase|storage-phase) phase_storage_phase ;;
  cstr|mixing) phase_cstr ;;
  soft)        phase_soft ;;
  fabrication) phase_fabrication ;;
  ensemble)    phase_ensemble ;;
  volmatch)    phase_volmatch ;;
  manifest)    phase_manifest ;;
  all)
    phase_tests
    phase_storage_phase
    phase_cstr
    phase_soft
    phase_fabrication
    phase_ensemble
    phase_manifest
    ;;
  help|*)
    cat <<'EOF'
Usage: ./run_rework.sh <phase>

Phases (independent, runnable in any order, idempotent via caches):
  tests           full pytest suite                    (~10 s)
  storage_phase   item 5 regression lock (rho_in)      (~3 s)
  cstr            item 1 — CSTR + linear sweep + Table 3 (~2–4 h)
  soft            item 2 — train soft + 4 tagged null checks (~3 h)
  fabrication     item 3 — diagnostic per surrogate/null check (~30 min)
  ensemble        item 4 — 10 seeds × full carbon axis (~3–4 h)
  volmatch        TASK V1 — volume-matched comparison   (~30 min)
  manifest        item 10 — manifest + figs + audits    (~30 min)
  all             every phase, in dependency order     (~10–12 h)

Env vars:
  CARBON_PRICE=80                primary policy carbon price
  WORKERS=10                     outer-parallelism for 06/07
  VAL_FRAC=0.05                  v1.5 E2 validation-sample fraction
  PRICES_AXIS="0 20 40 60 80 100 120 160"  ensemble price grid
  FAB_WINDOWS=12                 CoolProp-heavy windows per fabrication diagnostic
  MIXING_RESUME=0                0 recomputes mixing cells after hourly data fixes

Typical workflow:
  ./run_rework.sh tests          # 10 s — sanity check
  ./run_rework.sh cstr           # 3 h — Priority 0 item 1
  ./run_rework.sh soft           # 3 h — Priority 0 item 2
  ./run_rework.sh fabrication    # 30 min — Priority 0 item 3
  ./run_rework.sh ensemble       # 4 h — Priority 0 item 4
  ./run_rework.sh manifest       # 30 min — figures + manifest + audits
EOF
    ;;
esac
