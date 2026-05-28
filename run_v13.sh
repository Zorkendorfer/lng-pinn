#!/usr/bin/env bash
# v1.3 full rerun — ~3 hours unattended on M-series, with --workers 3.
#
# Key ordering change from prior versions: seed sensitivity now runs BEFORE
# the carbon sweep. The new fig6 design reads seed parquets to draw the
# ±2σ noise band; if seed runs after sweep, the band is missing from fig6
# on its first render.
set -e

# --- Step 0: kill any in-flight sweep / training ----------------------------
ps -eo pid,command | grep -E "07_carbon_sweep|06_seed_sensitivity|03_train_pinn|05_make_figures" \
    | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null || true

# --- Step 1: wipe stale caches from the buggy-singleton era -----------------
rm -f data/processed/train.parquet \
      data/processed/collocation_h_*.parquet \
      data/processed/true_costs_partial_*.parquet \
      data/processed/seed_sensitivity_*.parquet \
      data/processed/seed_sensitivity_*.json \
      data/processed/seed_true_costs_*.parquet
rm -f results/models/pinn_v1.pt results/models/pinn_v1.ckpt
rm -f results/tables/{dispatch_v1,baseline_,true_costs_,fidelity,sensitivity,surrogate_eval,yearly_summary,carbon_sweep_,seed_sensitivity}*

# --- Step 2: rebuild dataset on the fixed simulator (~5 min) ----------------
uv run python scripts/02_build_dataset.py

# --- Step 3: train v1.3 PINN with A1 cost loss (~30 min) --------------------
uv run python scripts/03_train_pinn.py --no-resume --lambda-c 1.0

# --- Step 4: seed sensitivity at €80, 5 seeds parallel (~60–90 min) ---------
#   Generates data/processed/seed_sensitivity_seed*.parquet, which fig6
#   reads to draw the ±2σ noise band. MUST run before step 5.
uv run python scripts/06_seed_sensitivity.py --no-resume --carbon-price 80 --workers 1

# --- Step 5: carbon sweep, 10 prices, 3 workers parallel (~40 min) ----------
#   Sweeps {0, 20, 40, 60, 70, 80, 90, 100, 120, 160} EUR/tCO2 — the dense
#   grid around the current EU ETS price (~€80) lets the figure resolve the
#   shape of the saving curve through the policy-relevant band.
#   Headline figure: results/figures/fig6_carbon_sweep.pdf (2-panel design
#   with seed-noise band, since step 4 populated the parquets).
uv run python scripts/07_carbon_sweep.py --workers 2 \
    --prices 0 20 40 60 70 80 90 100 120 160

# --- Step 6: dispatch + figs at €80/tCO2 (~20 min) --------------------------
#   Figs 1–5 with carbon-aware accounting; yearly_summary_true.csv.
uv run python scripts/04_run_dispatch.py --no-resume --carbon-price 80
uv run python scripts/05_make_figures.py --no-resume --carbon-price 80

echo "DONE. Paper-ready artifacts:"
echo "  results/figures/fig6_carbon_sweep.pdf       — headline (2-panel + seed band)"
echo "  results/figures/fig{1..5}_*.pdf              — supporting"
echo "  results/tables/carbon_sweep.csv             — saving vs CO2 price"
echo "  results/tables/seed_sensitivity_summary.csv — mean ± std at €80"
echo "  results/tables/yearly_summary_true.csv      — true-cost breakdown"
