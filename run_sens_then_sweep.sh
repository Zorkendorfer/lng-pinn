#!/usr/bin/env bash
# Sequential rerun of seed sensitivity then carbon sweep.
# Total wall-clock: ~2.2 hours on M-series with --workers 2.
#
# Use this after model + dispatch + train.parquet are valid but the seed
# sensitivity and carbon sweep need to be regenerated on extended data
# (e.g. after pulling new 2024-2025 prices).
#
# `set -e` aborts if seed sensitivity fails, so the carbon sweep doesn't
# kick off on a half-broken state.
set -e

echo "=== Phase 1: seed sensitivity (~1.5 hr) ==="
date
uv run python scripts/06_seed_sensitivity.py \
    --no-resume \
    --carbon-price 80 \
    --workers 2

echo
echo "=== Phase 2: carbon sweep (~40 min) ==="
date
uv run python scripts/07_carbon_sweep.py \
    --workers 2 \
    --prices 0 20 40 60 70 80 90 100 120 160

echo
echo "=== DONE ==="
date
echo "Artifacts:"
echo "  results/tables/seed_sensitivity_summary.csv"
echo "  results/tables/seed_significance.csv"
echo "  results/tables/carbon_sweep.csv"
echo "  results/figures/fig6_carbon_sweep.pdf"
echo "  results/figures/fig6_carbon_sweep_{2021,2022,2023,2024,2025}.pdf"
