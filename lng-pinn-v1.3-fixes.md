# LNG-PINN v1.3 — surrogate fidelity, carbon pricing, active learning

## Why v1.3

v1.2 produced a defensible negative result: composition-aware dispatch yields
~0% true-cost saving vs blind baselines on Lithuanian day-ahead prices, while
constant-flow loses 14–22% (temporal arbitrage dominates). v1.3 attacks the
two structural reasons that result is currently not publishable:

1. **The PINN can't be trusted on the dispatch trajectory.** Surrogate R² is
   0.9998 on a random LHS test split but the median *relative* cost error on
   the dispatch trajectory is ~97% (mean PINN cost €366/h vs CoolProp true
   €141/h on a 1000-point sample). The MSE-on-normalised-W loss does not
   penalise the relative cost error the dispatch cares about, and the LHS
   training distribution does not match the {10, 80} bimodal distribution
   the dispatch actually walks.
2. **The experiment has no signal for composition awareness to exploit.**
   `price_ch4_corr` mean ≈ 0.03 with std 0.27 across 156 weekly windows.
   Composition and day-ahead price are essentially uncorrelated by
   construction, so knowing the composition early can't change the optimal
   schedule meaningfully.

v1.3 is three fixes, ordered by effort and independence. A1 and B1 are
independent and can land in any order; A2 should land last because it
depends on A1 producing a well-conditioned surrogate.

---

## A1 — Cost-aware loss term

### Why

The dispatch objective is
`cost = price × W_total × m_dot × 3600 / 1000` (EUR/h).
The PINN currently minimises `mean((W_pred − W_true)²)` on normalised
outputs. That absolute MSE is dominated by the high-flow regime where
W_total is large in absolute terms — exactly the regime the dispatch
visits least. Low-flow points (m_dot=10) carry the smallest residuals
but the largest *relative* errors, and the cost formula amplifies them.

The relative cost error factors algebraically:
`(cost_pred − cost_true) / cost_true = (W_pred − W_true) / W_true`
— price and m_dot cancel. So a cost-aware loss collapses to a relative
MSE on the cost-relevant channels (W_pump, W_total). No price needed at
training time.

### Files

- `src/lng_pinn/pinn.py`
- `scripts/03_train_pinn.py`

### Changes

1. In `pinn.py`, add
   ```python
   def relative_cost_loss(
       y_pred_raw: Tensor,
       y_true_raw: Tensor,
       scaler: Scaler,
       eps: float = 1e-6,
   ) -> Tensor:
       """Relative-error MSE on W_pump and W_total (channels 0, 1).

       Aligns the training objective with the cost residual the dispatch uses:
           (cost_pred - cost_true) / cost_true = (W_pred - W_true) / W_true
       """
       y_pred = scaler.unscale_y(y_pred_raw)
       y_true = scaler.unscale_y(y_true_raw)
       w_pred = y_pred[:, [0, 1]]
       w_true = y_true[:, [0, 1]]
       return ((w_pred - w_true) / (w_true.abs() + eps)).pow(2).mean()
   ```
2. In `train()`, add `lambda_cost: float = 1.0` argument and append
   `loss_cost = relative_cost_loss(y_pred_d, yd, scaler)` to the total
   loss. The relative-cost term sits alongside the existing data MSE
   (NOT as a replacement — absolute and relative both matter).
3. Save `lambda_cost` in the resume checkpoint so a continued run keeps
   using the same loss weighting.
4. In `03_train_pinn.py`, add `--lambda-c` CLI flag (default 1.0).

### Acceptance criteria

After retraining from scratch with `--lambda-c 1.0`:
- Surrogate eval R² on the LHS test set stays > 0.99 on all four channels.
- A fresh fidelity table (run via `05_make_figures.py --fidelity-samples 1000`
  on a freshly built dispatch) shows median `|rel_error|` < 0.20 and
  p95 `|rel_error|` < 0.60. (Today: median 0.97, p95 5.05.)

### Risk

Low. Loss surgery only; no data, dispatch, or interface changes.

---

## B1 — Carbon-price term in dispatch

### Why

The aware-vs-blind delta is in the third decimal place because the only
thing tying composition to cost today is the small W_total differential
between compositions (~1–2%). Under EU ETS the carbon-cost differential
between LNG cargoes can be much larger — heavier hydrocarbons emit more
CO₂ per kg of fuel delivered. Adding a carbon-cost term to the dispatch
objective makes composition awareness economically meaningful in a way
that is realistic for European FSRUs operating under emissions pricing.

### Stoichiometry

Per kg of fuel:
- 1 mol CH₄ → 1 mol CO₂
- 1 mol C₂H₆ → 2 mol CO₂
- 1 mol C₃H₈ → 3 mol CO₂
- 1 mol n/iC₄H₁₀ → 4 mol CO₂
- N₂ inert

So
`mol_CO2_per_mol_fuel = x_CH4 + 2·x_C2H6 + 3·x_C3H8 + 4·(x_nC4 + x_iC4)`
`kg_CO2_per_kg_fuel  = mol_CO2_per_mol_fuel · MW_CO2 / sum(x_i · MW_i)`

For natural-gas compositions in the operating envelope this gives roughly
2.50–2.95 kg CO₂/kg fuel — a ~15% spread.

Carbon-cost per hour:
`co2_cost_eur_per_h = price_co2_eur_per_t · kg_CO2_per_kg_fuel(comp) · m_dot · 3600 / 1000`

Added to the existing `price × W_total × m_dot × 3600 / 1000` electricity-cost
term. The combined objective is what dispatch minimises.

### Files

- `src/lng_pinn/thermo.py` — add `co2_per_kg_fuel(x)`
- `src/lng_pinn/dispatch.py` — extend `optimize()` with `carbon_price_eur_per_t`
- `src/lng_pinn/baseline.py` — same extension on all 4 baselines
- `scripts/04_run_dispatch.py` — `--carbon-price` flag, default 0.0
- `scripts/06_seed_sensitivity.py` — same flag
- `scripts/05_make_figures.py` — new figure 6: carbon-price sweep

### Changes

1. `thermo.py`:
   ```python
   MW_CO2 = 44.009  # g/mol
   MW_SPECIES = {  # g/mol
       "Methane":   16.043,
       "Ethane":    30.070,
       "Propane":   44.097,
       "n-Butane":  58.123,
       "IsoButane": 58.123,
       "Nitrogen":  28.013,
   }
   _C_PER_MOL = (1, 2, 3, 4, 4, 0)  # carbons per molecule, aligned with SPECIES order

   def co2_per_kg_fuel(x: tuple[float, ...]) -> float:
       """kg of CO2 released per kg of fuel fully combusted (no slip, no flare)."""
       mol_co2 = sum(xi * ci for xi, ci in zip(x, _C_PER_MOL))
       mw_fuel = sum(xi * MW_SPECIES[sp] for xi, sp in zip(x, SPECIES))
       return mol_co2 * MW_CO2 / mw_fuel
   ```
2. `dispatch.optimize(...)` gains `carbon_price_eur_per_t: float = 0.0`. The
   per-hour cost expression becomes
   ```
   electricity_cost[t] + carbon_price_eur_per_t * co2_factor * m_dot[t] * 3.6
   ```
   (3600 / 1000 = 3.6). `co2_factor` is a single scalar per window because
   composition is constant inside the optimisation window. Mixed-composition
   windows interpolate as today.
3. Same change in `baseline.optimize_blind_*` and `optimize_constant_flow`.
4. CLI flag `--carbon-price` (units EUR/tCO₂, default 0.0) plumbed through
   04 and 06. The value is recorded in a new top-level row in the output
   parquets — or as filename suffix — so multiple sweeps coexist.
5. `05_make_figures.py` gains `fig6_carbon_sweep`. The sweep runs in a
   loop: for `price_co2 ∈ {0, 20, 40, 80, 120, 160}`, call dispatch via a
   programmatic entry point, record `saving_vs_horizon_pct` per year per
   price. Output: PDF showing `saving_pct` vs `price_co2` with one line
   per year (or a single mean ± std band across years).

### Acceptance criteria

- `co2_per_kg_fuel` against hand-calculated reference for pure CH₄ (2.74),
  TYPICAL_LNG (~2.78), and a heavy mix (~2.95) — unit test in
  `tests/test_thermo.py`.
- With `--carbon-price 0`, dispatch results are bit-identical to v1.2.
- With `--carbon-price 80`, `saving_vs_horizon_pct` in
  `yearly_summary_true.csv` is > 0.5% in at least 2 of 3 years.
- Carbon-price sweep PDF shows monotonic increase of saving with price.

### Risk

Medium. Touches dispatch and 4 baselines; needs the same change in all
five to keep the comparison fair. Backward-compat is preserved by the
default `0.0` value.

---

## A2 — Active-learning trajectory augmentation

### Why

The training set is a uniform LHS over the operating envelope. The
dispatch concentrates on m_dot ∈ {10, 80} (bang-bang at the bounds) and
on the actual cargo-cycle composition trajectory. The PINN is fit to
the wrong distribution. A2 closes that gap by **labelling the actual
dispatch trajectory with CoolProp ground truth and adding those points
to the training set**. One pass is enough.

A2 depends on A1 because the trajectory generated by an uncorrected PINN
would itself be skewed by the cost bias — augmenting on bad trajectories
reinforces the bias. With A1 in place the trajectory points are sensible
choices that the model needs to be calibrated on.

### Files

- New script `scripts/03b_augment_dataset.py`
- `src/lng_pinn/dataset.py` — small helper to append rows with a
  `_source` tag and re-deduplicate.

### Changes

1. `03b_augment_dataset.py`:
   ```python
   def main():
       # 1. Load model + timeseries
       # 2. Run full dispatch on aware strategy only (reusing 04 code path
       #    via a programmatic entry).
       # 3. Extract unique (composition, m_dot, T_amb, T_sw) rows from the
       #    dispatch output. Round m_dot/T to 4 dp to dedupe near-duplicates.
       # 4. CoolProp-label them in parallel via dataset._simulate_one.
       # 5. Append to data/processed/train.parquet with _source="trajectory".
       # 6. Print summary: rows added, rows skipped (CoolProp failure or
       #    already in train set within tolerance).
   ```
2. `dataset.py` gains `append_trajectory_rows(rows: list[dict]) -> int`
   that handles the dedupe and tagging logic.
3. The training set gains an optional `_source` column. `03_train_pinn.py`
   does not need to know about it; the scaler and split are unchanged.

### Workflow

```
02_build_dataset.py            # LHS samples, 20k rows
03_train_pinn.py --no-resume   # train v1 (with A1 + A2-friendly losses)
04_run_dispatch.py             # produces dispatch_v1.parquet
03b_augment_dataset.py         # appends ~5–10k trajectory rows
03_train_pinn.py --no-resume   # train v2 on augmented set
04_run_dispatch.py --no-resume # produces dispatch_v2.parquet
05_make_figures.py --no-resume # fresh true_costs + fidelity for v2
```

### Acceptance criteria

- After A1+A2, median `|rel_error|` on fidelity < 0.05 and p95 < 0.20.
- Aware-vs-horizon true-cost saving at `--carbon-price 80` is within ±0.3
  percentage points of the same number computed from a CoolProp-in-the-loop
  reference (sample 200 random windows, full CoolProp dispatch).

### Risk

Medium-high. Adds a manual two-pass workflow. Easy to forget the second
training run, so the script prints a clear "now rerun 03 and 04" message.

---

## Carbon-price sweep figure (depends on B1)

### Goal

The headline figure of the paper. Shows the saving from composition
awareness as a function of carbon price.

### Sweep

`price_co2 ∈ {0, 20, 40, 80, 120, 160} EUR/tCO₂` (6 points).

For each price, run a full dispatch + 5 baselines on the 3-year timeseries.
Persist per-price `yearly_summary_true.csv` to
`results/tables/carbon_sweep_co2_<price>.csv`.

### Plot

X-axis: `price_co2`. Y-axis: `saving_vs_horizon_pct`. One line per year +
mean line. Annotate the €80/tCO₂ point (current EU ETS).

### Effort

~2 hours once B1 is in. The dispatch is already cached/resumable so
adding price as a sweep dimension is just a wrapper script.

---

## Expected outcomes after v1.3

| Metric | v1.2 | v1.3 target |
|---|---|---|
| Fidelity median rel_error | 0.97 | < 0.05 |
| Fidelity p95 rel_error | 5.05 | < 0.20 |
| Aware vs horizon (no carbon) | 0 ± 0.1 % | unchanged |
| Aware vs horizon (@€80/tCO₂) | n/a | 1–3 % robust |
| Aware vs constant (@€80/tCO₂) | 14–22 % | 18–28 % |

## Paper conclusion after v1.3

> "Composition-aware dispatch becomes economically meaningful under EU ETS
> carbon pricing. Without carbon pricing, temporal price arbitrage is the
> dominant lever and composition information is operationally irrelevant.
> The PINN-MILP framework solves the joint problem efficiently, with
> trajectory-aware training reducing surrogate cost error by 20× compared
> to LHS-only training. Savings scale approximately linearly with carbon
> price; below ~€30/tCO₂ the benefit is below operational noise."

This is a publishable positive result with a clear practical message:
*under what conditions does composition-aware dispatch pay off*.

---

## Implementation order

1. **A1** — drop-in loss change (~2 hrs, 1 commit). Re-run 03 and 04 with
   `--no-resume` to regenerate dispatch on the corrected model.
2. **B1** — carbon-pricing in dispatch + baselines + tests (~3 hrs, 1 commit).
   Re-run 04 with `--carbon-price 80` to validate.
3. **Carbon-price sweep figure** (~2 hrs, 1 commit). Wrapped around 04 +
   05; produces fig6.
4. **A2** — active learning script + two-pass workflow (~1 day, 2 commits).
   Final dispatch + figures regenerated.

Each step is independently verifiable. Estimated total wall-clock: 2 days
of focused work.
