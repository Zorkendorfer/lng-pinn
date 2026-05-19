# LNG-PINN v1.2 — fix plan

Three root causes explain why v1.1 savings (0.68% ± 1.56%) are indistinguishable from noise.
Fix them in order — each one is a prerequisite for the next.

---

## Root cause 1 — Baseline is too strong

`optimize_blind_horizon` uses the mean composition of the **exact same 7-day window**.
That composition is already ~95% correlated with the hour-by-hour actual composition
(cargoes change every 12 days; the window is only 7 days). There is almost no information
gap for the aware strategy to exploit.

### Fix 1: lagged-composition blind baseline

Replace the horizon-mean baseline with a **lagged-cargo baseline**: the blind strategy
uses the composition of the *previous* cargo (i.e., the last archetype transition that
completed before the window starts). This is the realistic "operator knows last week's
cargo but not today's blend" assumption.

**Files:** `src/lng_pinn/baseline.py`, `scripts/04_run_dispatch.py`,
`scripts/06_seed_sensitivity.py`

**Change:**
```python
def optimize_blind_lagged(
    horizon_df: pd.DataFrame,
    model: PINNMLP,
    scaler: Scaler,
    demand_kg: float,
    lagged_composition: pd.Series,   # composition at start of window, not window mean
    inv0: float = 0.5,
) -> Schedule:
    blind_df = _with_fixed_composition(horizon_df, lagged_composition)
    return optimize(blind_df, model, scaler, demand_kg, inv0)
```

In `04_run_dispatch.py`, derive `lagged_composition` as the composition at `ts.iloc[start]`
(the first hour of the window, not the mean). This represents "operator uses current
measured composition, unchanged for the whole horizon."

Keep `optimize_blind_horizon` as a second baseline (it is still a valid comparison point).
Headline table then has 5 rows: aware, blind-lagged, blind-horizon, blind-annual, constant.

**Done when:** lagged baseline parquet exists; seed sensitivity re-run shows aware vs
lagged saving is larger and less noisy than aware vs horizon saving.

---

## Root cause 2 — Cost comparison uses PINN predictions, not true costs

All recorded `cost_eur` values are PINN-predicted. Strategies using different composition
assumptions (annual mean, lagged, horizon mean) see different PINN "prices" for identical
physical operations. This creates a systematic bias: whichever composition assumption
gives lower W_total in the PINN wins, regardless of true physics.

Proof: blind-annual records 4.8M EUR vs aware's 9.5M EUR — a 50% difference that has
no physical basis and is entirely a PINN generalisation artefact.

### Fix 2: evaluate all strategies against CoolProp true costs

After dispatch (which uses PINN for optimisation), re-evaluate every strategy's
resulting `m_dot` schedule through the CoolProp plant simulator to get true `W_total`
and true `cost_eur`. Compare only these true costs.

**Files:** `scripts/05_make_figures.py`

**Change:** extend `build_fidelity_table` to accept a dict of strategy DataFrames and
produce per-strategy true costs. Replace `build_yearly_summary` inputs with true-cost
DataFrames rather than PINN-cost DataFrames.

```python
def build_true_cost_df(
    dispatch_df: pd.DataFrame,   # has columns: time, m_dot
    ts_df: pd.DataFrame,          # has: price_eur_mwh, T_amb, T_sw, composition cols
) -> pd.DataFrame:
    """Re-evaluate m_dot schedule through CoolProp; return df with true_cost_eur."""
    ...
```

Save as `results/tables/true_costs_{strategy}.parquet` for each strategy.
`yearly_summary.csv` uses true costs only.

**Done when:** yearly_summary.csv contains `_true_eur` columns; annual baseline no
longer shows -97% artefact.

---

## Root cause 3 — Composition signal too weak for the current operating envelope

Even with a lagged baseline and true costs, savings may be small because:
- W_total varies only ~3-8% across the operating composition range
- Electricity price varies ~200% — price timing dominates the optimisation
- Within-window composition variation is small (5-day blend, not step change)

This is not a bug — it is a finding. The paper should report it honestly and explain
*when* composition awareness does pay off.

### Fix 3a: widen dataset composition range

The current LHS sampling already covers the full BOUNDS range. But the bounds for
heavy components (C3H8, nC4, iC4) are narrow. Widen to include more LNG-rich cargoes
where W_trim varies more:

```python
BOUNDS = {
    "CH4":    (0.78, 0.96),   # was (0.82, 0.96)
    "C2H6":   (0.02, 0.15),   # was (0.02, 0.12)
    "C3H8":   (0.005, 0.060), # was (0.005, 0.035)
    "nC4H10": (0.001, 0.025), # was (0.001, 0.015)
    "iC4H10": (0.001, 0.015), # was (0.001, 0.010)
}
```

Also add two richer archetypes to `composition.py`:
```python
"Nigeria":   (0.860, 0.080, 0.040, 0.015, 0.005, 0.000),
"Australia": (0.875, 0.085, 0.025, 0.010, 0.004, 0.001),
```

**Files:** `src/lng_pinn/dataset.py`, `src/lng_pinn/composition.py`

Rebuild dataset: `uv run python scripts/02_build_dataset.py`
Retrain PINN: `uv run python scripts/03_train_pinn.py --steps 100000 --patience 10000`

### Fix 3b: regime-conditional analysis in paper

Add a column `price_vol_regime` (low/medium/high tercile of price volatility) to the
sensitivity table. Show that aware vs lagged saving is ~0% in low-volatility and
positive (target >1.5%) in high-volatility regimes. This is the honest story.

**Files:** `src/lng_pinn/plots.py` — `fig_sensitivity` already computes `price_volatility`;
add regime grouping.

**Done when:** sensitivity figure shows a clear positive slope of saving vs
price_volatility; the regime-conditional table exists.

---

## Suggested execution order

| Step | Action | Blocks |
|------|--------|--------|
| 1 | Implement Fix 1 (lagged baseline) | — |
| 2 | Implement Fix 2 (true cost eval) | — |
| 3 | Re-run dispatch: `uv run python scripts/04_run_dispatch.py` | Fix 2 output |
| 4 | Re-run figures: `uv run python scripts/05_make_figures.py` | Fix 2 output |
| 5 | Re-run seed sensitivity: `uv run python scripts/06_seed_sensitivity.py` | Fix 1 output |
| 6 | Rebuild dataset with wider bounds (Fix 3a) | Retrain |
| 7 | Retrain: `uv run python scripts/03_train_pinn.py --steps 100000 --patience 10000` | Steps 3-5 |
| 8 | Re-run steps 3-5 with new model | — |
| 9 | Add regime-conditional figure (Fix 3b) | Step 8 |

Steps 1-5 can be done without rebuilding the dataset and give an honest picture of v1.1
savings under a fair baseline and true costs. Do these first. If savings are still <1%
with the lagged baseline, Fix 3a (wider dataset) is the only remaining lever.

---

## Definition of done (v1.2)

1. `baseline.py` has `optimize_blind_lagged`.
2. `05_make_figures.py` builds true-cost DataFrames via CoolProp for all strategies.
3. `yearly_summary.csv` uses true costs; annual baseline shows a physically plausible number.
4. Seed sensitivity with lagged baseline: mean saving clearly positive (>0.5%) with
   std < mean, OR the analysis honestly concludes savings are noise-level and the paper
   is reframed accordingly.
5. `fig2_sensitivity.pdf` shows saving vs price volatility with positive slope.
6. All scripts run end-to-end from a clean state in under 2 hours on an M-series Mac.
