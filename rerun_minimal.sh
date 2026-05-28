#!/usr/bin/env bash
# v1.3 minimal rerun — ~1 hour on M-series.
#
# Use this when train.parquet, pinn_v1.pt, dispatch_v1.parquet, the 6
# existing carbon-sweep CSVs, and figs 1–5 are all valid. Only refreshes
# the seed sensitivity (which was still showing pre-singleton-fix numbers)
# and adds the 4 new sweep prices (60/70/90/100), then re-renders fig6
# with the now-fresh seed noise band.
#
# If anything upstream of seed sensitivity is stale, run ./run_v13.sh
# instead.
set -e

# --- Step 1: wipe ONLY the stale seed sensitivity artifacts -----------------
#   train.parquet, the model, and dispatch results stay untouched.
rm -f data/processed/seed_sensitivity_*.parquet \
      data/processed/seed_sensitivity_*.json \
      data/processed/seed_true_costs_*.parquet \
      results/tables/seed_sensitivity*.csv

# --- Step 2: seed sensitivity at €80 (~45 min, all 12 cores per seed) -------
#   Produces data/processed/seed_sensitivity_seed*.parquet — the source
#   plots.py:fig_carbon_sweep reads to draw the ±2σ noise band.
uv run python scripts/06_seed_sensitivity.py --no-resume --carbon-price 80 --workers 1

# --- Step 3: sweep only the 4 new prices (~15 min) --------------------------
#   The per-price CSV cache makes 07_carbon_sweep.py skip the existing
#   {0, 20, 40, 80, 120, 160}. It still calls fig_carbon_sweep at the end,
#   which re-renders fig6 with the fresh noise band from step 2.
uv run python scripts/07_carbon_sweep.py --workers 2 --prices 60 70 90 100

# --- Step 4: belt-and-braces fig6 re-render ---------------------------------
#   If step 3 found all prices already cached it still calls fig_carbon_sweep,
#   but if it skipped that for any reason this guarantees fig6 reflects the
#   full 10-price grid + fresh noise band.
uv run python -c "
import pandas as pd
from lng_pinn.plots import fig_carbon_sweep
sweep = pd.read_csv('results/tables/carbon_sweep.csv')
fig_carbon_sweep(sweep)
print('fig6_carbon_sweep.pdf re-rendered with full 10-price grid + seed noise band')
"

echo "DONE. Refreshed artifacts:"
echo "  results/figures/fig6_carbon_sweep.pdf       — headline w/ noise band"
echo "  results/tables/carbon_sweep.csv             — 10-price grid"
echo "  results/tables/seed_sensitivity_summary.csv — refreshed v1.3 numbers"
