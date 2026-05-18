# LNG-PINN v1.1 — fix plan toward arXiv submission

A focused plan to fix the gaps between v1 and what's defensible as a "physics-informed" paper. Read after `lng-pinn-plan.md`. Same rule as before: each task has a binary done condition; don't move on until it's ticked.

The goal is to **close the gap between the paper's claims and what the code actually does**, not to add new capabilities. Phase order matches dependency order; A and B can be done in parallel.

---

## 0. Anti-scope (read first, re-read weekly)

You will be tempted to expand the project while fixing it. Refuse. The following are **out of scope** for v1.1:

- Adding transient regasification — still steady-state.
- Adding cargo scheduling as a decision variable — composition stays exogenous.
- Adding intraday or balancing markets — day-ahead only.
- Adding gas-quality (Wobbe/methane number) curtailment constraints — these are v2 hooks for the discussion section, nothing more.
- Multi-FSRU or other terminals — still Independence, Klaipėda.
- Replacing the PINN with a different architecture (Fourier features, SIREN, transformer, NODE) — keep the MLP.
- Hyperparameter sweep beyond what the plan already covers.
- Adding a new ML framework (JAX, Equinox, Lightning) — pure PyTorch.
- Rewriting any module "for cleanliness" beyond what's specified below.

If a task here motivates an out-of-scope idea, write the idea in `paper/v2-ideas.md` and move on.

---

## 1. What v1.1 must fix (one-paragraph diagnosis)

The v1 code runs end-to-end and produces reproducible numbers, but four claims in the paper are not fully supported by the implementation:

1. **"Physics-informed"** — the current physics loss is `relu(-W_pump) + relu(W_pump - W_total)`, which is a sign/ordering penalty on outputs, not an energy balance residual.
2. **"Cost reduction from composition-aware dispatch"** — the headline is 0.74% over 3 years, with 2023 negative. The plant model has `W_total` independent of `m_dot`, which collapses dispatch to bang-bang and limits the mechanism by which awareness can pay off. The baseline definition also differs from the plan ("horizon-mean" vs the planned "annual-mean").
3. **"Surrogate fidelity within Y%"** — currently 68% within ±1%, with a small fraction of samples predicting unphysical negative `W_total`. There is no train/val/test split on the surrogate itself.
4. **"Composition variability drives savings"** — the data falsifies this (correlation −0.08, highest-variability tercile is *negative*). The paper must either re-frame this as a negative result or expose a different mechanism.

v1.1 fixes (1), (3), and the baseline ambiguity in (2). The flow-dependence in (2) is partially addressed (Phase B). (4) is a framing change, not a code change.

---

## 2. Phase A — PINN physics fixes

### A1. Replace the fake "physics" loss with a real energy balance residual

**Files:** `src/lng_pinn/pinn.py`, `src/lng_pinn/dataset.py`, `src/lng_pinn/thermo.py`.

**Change:**

1. In `src/lng_pinn/dataset.py`, extend `_simulate_one` to also store `h_in_per_kg` and `h_out_per_kg`:

   ```python
   # In _simulate_one, after the existing simulate() call:
   from lng_pinn.thermo import get_state
   import CoolProp.CoolProp as CP

   state = get_state(x)
   state.update(CP.PT_INPUTS, 1.0e5, 111.0)  # storage state
   h_in_per_kg = state.hmolar() / state.molar_mass()
   state.update(CP.PT_INPUTS, 80.0e5, 278.15)  # send-out state
   h_out_per_kg = state.hmolar() / state.molar_mass()
   ```

   Add `h_in_per_kg` and `h_out_per_kg` to the returned dict and the schema docstring of `build_training_set`.

2. In `src/lng_pinn/pinn.py`, replace `energy_balance_residual` entirely:

   ```python
   ETA_TRIM_HEATER = 0.98  # match plant.py

   def energy_balance_residual(
       x_raw: Tensor,
       y_pred_raw: Tensor,
       scaler: Scaler,
       h_in: Tensor,    # (B,) J/kg
       h_out: Tensor,   # (B,) J/kg
   ) -> Tensor:
       """Steady-state energy balance: h_out - h_in == W_pump + W_trim*eta + Q_sw.

       Q_sw must be non-negative (seawater is a heat source, not sink).
       """
       y = scaler.unscale_y(y_pred_raw)
       W_pump = y[:, 0] * 3.6e6   # kWh/kg -> J/kg
       W_total = y[:, 1] * 3.6e6
       W_trim = W_total - W_pump
       W_trim_heat = W_trim * ETA_TRIM_HEATER
       delta_h = h_out - h_in
       Q_sw_implied = delta_h - W_pump - W_trim_heat
       # Normalise by typical |delta_h| ~ 5e5 J/kg for scale-invariance
       return torch.relu(-Q_sw_implied / 5e5).pow(2).mean()
   ```

3. Update `scripts/03_train_pinn.py` to pass `h_in` and `h_out` columns into the training loop alongside `X_col`. Store them as torch tensors aligned to the collocation index.

**Done when:**
- `tests/test_pinn_residuals.py::test_energy_residual_below_threshold` passes with the new residual on a freshly trained checkpoint.
- A new test `test_energy_balance_holds_on_training_data` checks that on 1000 random training-set samples, `|delta_h - W_pump - W_trim*eta - Q_sw_true| / delta_h < 0.02` for the *plant simulator* (sanity check that the residual formula matches the simulator).

### A2. Add an analytical pump-work residual

**File:** `src/lng_pinn/pinn.py`.

**Change:** Add a second physics term — pump work is determined by composition + pressures, so the PINN's predicted `W_pump` should match the analytical form:

```
W_pump_expected = (P_out - P_in) / rho_in(comp) / eta_pump
```

Pre-compute `W_pump_expected_per_kg` as a new column in `train.parquet` in `dataset.py` (analytic from `rho_in` returned by CoolProp). Then add:

```python
def pump_work_residual(y_pred_raw, scaler, W_pump_expected):
    y = scaler.unscale_y(y_pred_raw)
    return ((y[:, 0] - W_pump_expected) / W_pump_expected).pow(2).mean()
```

Loss becomes `lambda_d * L_data + lambda_e * L_energy + lambda_p * L_pump`.

**Done when:** trained PINN has MAE on `W_pump` < 1% on a held-out test set (A4 below).

### A3. Softplus output transform on W_total to prevent negative predictions

**File:** `src/lng_pinn/pinn.py`.

**Change:** Wrap the W_total output channel in a softplus so it can never go negative:

```python
class PINNMLP(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        # ... (existing layers) ...
        self.softplus = nn.Softplus(beta=10.0)

    def forward(self, x: Tensor) -> Tensor:
        raw = self.net(x)
        # Channels: [W_pump, W_total, T_out, exergy]
        # Apply softplus to W_pump and W_total (must be non-negative).
        # Apply softplus to exergy (must be non-negative).
        # Leave T_out raw (it's normalised by scaler anyway).
        ...
```

Note that this is applied in normalised space, so be careful with the scaler. The simplest correct version: apply softplus *after* unscaling to physical units, not in scaled space. Pick a convention and document it.

**Done when:** `test_w_total_positive_on_random_inputs` passes with `negative_frac == 0.0` (tighten from 5% to 0%). `min(fidelity_df["pinn_cost_eur"]) >= 0` on the post-fix fidelity table.

### A4. Held-out train/val/test split on the surrogate

**File:** `scripts/03_train_pinn.py`, `src/lng_pinn/pinn.py`.

**Change:**
- Split `train.parquet` 80/10/10 with a fixed seed.
- Train on the 80% split; early-stop on val MSE (patience = 2000 steps); evaluate on the 10% test split at end of training.
- Save a `results/tables/surrogate_eval.parquet` with per-channel MAE, RMSE, R² on the test split.

**Done when:** `surrogate_eval.parquet` exists; the paper's Methods section can quote per-channel test errors with a clean train/val/test discipline.

### A5. Real collocation points

**File:** `scripts/03_train_pinn.py`.

**Change:** Replace the current `col_np = X_np[rng.choice(...)]` with a fresh Latin hypercube sample over the same `BOUNDS` dictionary used in `dataset.py`. Composition rows must still sum to 1 (use the existing `_sample_compositions` helper). The collocation set has *no* CoolProp ground truth — that's the point. Size ~10,000 is fine.

**Done when:** collocation points span the full input bounding box and are *not* subsampled from training data. A test checks that fewer than 1% of collocation points exactly coincide with a training point.

---

## 3. Phase B — Plant model realism

The current plant model has `W_total` independent of `m_dot`, which makes cost linear in flow and the dispatch policy bang-bang. That collapses the value of any kind of dispatch optimisation. v1.1 adds a single, defensible flow-dependent term.

### B1. Flow-dependent pump efficiency

**File:** `src/lng_pinn/plant.py`.

**Change:** Replace the constant `ETA_PUMP = 0.75` with a quadratic efficiency curve around a best-efficiency point (BEP):

```python
ETA_PUMP_BEP = 0.78
M_DOT_BEP = 45.0   # kg/s
ETA_PUMP_CURVATURE = 8e-5  # (kg/s)^-2

def pump_efficiency(m_dot: float) -> float:
    """Quadratic efficiency curve around the best-efficiency point.

    Conservative literature values: large cryogenic pumps lose ~10–15
    efficiency points at 25% and 150% of BEP. See e.g. Karassik (Pump
    Handbook), Chapter 2.
    """
    return ETA_PUMP_BEP - ETA_PUMP_CURVATURE * (m_dot - M_DOT_BEP) ** 2
```

Use this in `simulate()` instead of the constant. Add `pump_efficiency` to the simulate signature implicitly via `m_dot`.

**Important:** also update `dataset.py` to include `m_dot` in the LHS for collocation/training even after the change (already there, but verify).

**Done when:** `test_higher_flow_more_total_work` is renamed `test_pump_eta_varies_with_flow` and asserts that `simulate(comp, 10, ...).W_pump > simulate(comp, 45, ...).W_pump` and `simulate(comp, 80, ...).W_pump > simulate(comp, 45, ...).W_pump` (U-shape).

### B2. Optional: trim heater turndown penalty

**File:** `src/lng_pinn/plant.py`.

**Change:** Add a small penalty to `ETA_TRIM_HEATER` at very low flow to reflect heat-loss-to-ambient becoming a larger fraction of throughput. Skip if time-constrained — B1 alone is enough to break the bang-bang regime.

```python
def trim_heater_efficiency(m_dot: float) -> float:
    # ~95% at min turndown, asymptotes to 98% at high flow
    return 0.98 - 0.03 * (M_DOT_MIN / max(m_dot, M_DOT_MIN)) ** 2
```

**Done when:** completed or explicitly skipped with a one-line note in the commit message.

### B3. Plant tests reflect the new physics

**File:** `tests/test_plant.py`.

**Change:**
- Add `test_w_total_varies_with_flow`: assert `simulate(comp, 10).W_total != simulate(comp, 45).W_total != simulate(comp, 80).W_total` (relative difference > 1%).
- Add `test_energy_balance_closes_within_half_percent`: compute `delta_h_required` from CoolProp, compute `W_pump + W_trim*eta + Q_sw`, assert they match within 0.5%.

**Done when:** both tests pass.

---

## 4. Phase C — Dispatch and baselines

### C1. Add a true annual-mean baseline

**File:** `src/lng_pinn/baseline.py`, `scripts/04_run_dispatch.py`.

**Change:** Keep the current horizon-mean baseline but rename it `optimize_blind_horizon`. Add `optimize_blind_annual` that uses a single composition computed once across the full timeseries (not per-window). The plan promised the annual version; reviewers will ask why it's not there.

In `04_run_dispatch.py`, run both baselines and save two parquets (`baseline_horizon_v1.parquet`, `baseline_annual_v1.parquet`).

**Done when:** both baseline parquets exist; figures and tables report savings against both, side by side.

### C2. Rolling horizon should actually roll

**File:** `scripts/04_run_dispatch.py`.

**Change:** Change `step = H` to `step = 24`. With 7-day horizon and daily step, you get rolling-horizon receding-horizon dispatch — keep only the first 24 hours of each optimisation, then slide forward. This is the standard MPC-style backtest.

You'll need to:
- Carry inventory forward between windows (current code resets `inv0` per window).
- Only record the first 24 hours of each window in the output parquets.

**Done when:** dispatch result has 24 × (n_days − 6) rows ≈ 3 × 365 days × 24 hours ≈ 26k rows (similar size to current); inventory is continuous across windows; total cost is computed correctly.

### C3. Add a no-information constant-flow baseline (sanity)

**File:** `src/lng_pinn/baseline.py`.

**Change:** Add `optimize_constant_flow(horizon_df, demand_kg)` that simply dispatches at a constant `m_dot = demand_kg / (H * 3600)` for all hours. No optimisation. This is the lower bound — if your "smart" methods don't beat constant flow by a clear margin, you have no story.

**Done when:** constant-flow baseline runs and its total cost is reported alongside the others. Headline table has 4 rows: aware, blind-horizon, blind-annual, constant.

---

## 5. Phase D — Evaluation and figures

### D1. Yearly results table

**File:** `scripts/05_make_figures.py`, `src/lng_pinn/plots.py`.

**Change:** Add `results/tables/yearly_summary.csv` with columns `year, aware_eur, blind_horizon_eur, blind_annual_eur, constant_eur, saving_vs_horizon_pct, saving_vs_annual_pct, saving_vs_constant_pct`.

**Done when:** the table exists; 2023's negative-vs-horizon result is preserved (you cannot hide this).

### D2. Composition seed sensitivity

**File:** new `scripts/06_seed_sensitivity.py`.

**Change:** Re-run the dispatch backtest with composition seeds {42, 0, 1, 7, 13}. Report mean ± std of yearly saving across seeds. This is the most important robustness check — if the result is seed-dependent, you need to know.

**Done when:** seed-sensitivity table in `results/tables/seed_sensitivity.csv`. Add a sentence to the abstract's headline number with the seed-averaged value if the spread is meaningful.

### D3. Fidelity table on held-out test set, in addition to dispatch-points

**File:** `scripts/05_make_figures.py`.

**Change:** In addition to the existing Fig 4 (fidelity on dispatch outputs), add a second fidelity panel or sub-table for the held-out 10% LHS test set from A4. Report MAE on W_pump, W_total, T_out, exergy separately. This is the surrogate paper's standard report.

**Done when:** `results/tables/surrogate_eval.parquet` from A4 is consumed by `make_figures.py` and either a new figure or a clearly numbered table is produced.

### D4. Sensitivity figure (Fig 2) redux

**File:** `src/lng_pinn/plots.py`.

**Change:** The current Fig 2 (cost saving vs std(CH4)) has correlation ≈ 0 and tells a negative story. Keep the scatter but also add: (a) a sub-panel of saving vs *price volatility* in the same window, and (b) a sub-panel of saving vs *price–composition correlation* in the window. One of these will likely be the actual driver of when awareness pays off.

**Done when:** Fig 2 has either two side-by-side panels or three; the correlation with the best predictor is annotated on the chart.

---

## 6. Phase E — Paper text updates

These are short and mechanical; do them last, after the numbers are stable.

### E1. Honest abstract

**File:** `paper/main.tex`.

Replace the X/Y/Z placeholders with actual numbers from v1.1 results. If the headline is still ~1%, the abstract framing should be: "We present an end-to-end physics-informed surrogate + dispatch framework. Composition awareness yields ~X% cost reduction over a strong (horizon-mean) baseline and ~Y% over an annual-mean baseline, with the benefit concentrated in high-volatility price regimes."

### E2. Methods section reflects the *actual* physics loss

**File:** `paper/main.tex` §3.2.

Write out the actual energy-balance residual implemented in A1, including the `Q_sw ≥ 0` and pump-work residuals. Cite Wang et al. 2021 only if you actually implement gradient-norm balancing (still optional and out of scope unless trivial).

### E3. Discussion includes the negative result

**File:** `paper/main.tex` §6.

Explicitly state: composition variability does *not* predict cost savings; the value of awareness is regime-conditional. Hypothesise the actual driver (Phase D4 result). List the v2 directions: intraday market, gas-quality curtailment, cargo schedule as decision variable, transient regasification.

### E4. Limitations subsection

**File:** `paper/main.tex` §6.

Add a clear bulleted limitations subsection:
- Steady-state regasification; no ramp/start-up dynamics.
- Synthetic cargo compositions from GIIGNL archetypes; no real Klaipėda cargo schedule.
- Day-ahead market only; no intraday or balancing market.
- Pump efficiency curve is a literature value, not plant-fitted.
- Single FSRU; results may not generalise to onshore terminals.

---

## 7. Suggested execution order

Each phase has rough effort estimates assuming you're working alone in evenings/weekends.

| Order | Phase | Tasks | Effort | Can run in parallel? |
|---|---|---|---|---|
| 1 | A | A1, A2, A3, A4, A5 | 6–10h | Internally sequential |
| 2 | B | B1, B3 (skip B2 unless time) | 2–3h | Parallel to A |
| 3 | (retrain) | Re-run scripts 02 → 03 | 1–2h wall | Blocks C–E |
| 4 | C | C1, C2, C3 | 4–6h | Internally parallel |
| 5 | (re-backtest) | Re-run script 04 | 0.5h | Blocks D |
| 6 | D | D1, D2, D3, D4 | 6–10h | D2 is the long pole |
| 7 | E | E1, E2, E3, E4 | 3–5h | Sequential |

Total: ~25–40 hours of focused work. **Two to three weekends, not two days.** Do not try to compress this before Wednesday with Khan — present v1 honestly and discuss the fix plan with him.

---

## 8. Definition of done (v1.1)

v1.1 is ready for arXiv when:

1. All Phase A done conditions hold.
2. Plant model has flow-dependent pump efficiency; dispatch is no longer near-bang-bang (intermediate flow levels chosen in ≥15% of hours).
3. Three baselines reported (horizon-mean, annual-mean, constant) with side-by-side savings.
4. Rolling horizon is actually rolling.
5. Seed-sensitivity table exists and seed-averaged numbers appear in the abstract.
6. Paper's Methods section describes the *actual* physics loss implemented.
7. Paper's Discussion includes the negative variability-savings result.
8. `make reproduce` (or equivalent shell script) takes a fresh clone to all figures in `results/figures/` within ~1 hour on an M-series Mac.
9. arXiv preprint posted; GitHub tagged `v1.1.0`; Zenodo DOI minted; README links to both.

If at any point you find yourself adding to the anti-scope list (Section 0), close the laptop and re-read Section 0 again.
