# LNG-PINN v1.5 — Phase-2 performance + cross-seed caching

## Why v1.5

v1.4 established the publishable result: composition-aware FSRU dispatch saves
**+2.50% ± 0.79% (10 seeds × 5 years, p < 1e-5)** over a lagged-composition
baseline at €80/tCO₂. The pipeline takes ~8 hours wall-clock end-to-end on
M-series CPU (5-year LT data, 10 seeds, 10 carbon prices). Running the same
analysis for Germany or any other ENTSO-E zone would be ~10–12 hours.

v1.5 is pure engineering work — no methodology change, no new metrics — to
make the cost-table-and-cost-eval bottleneck fast enough that "let's also do
Germany" is a 1–2 hour decision rather than a 2-day commitment. Three
orthogonal optimisations, each with a defensible accuracy contract.

---

## E1 — CoolProp dedupe-before-submit (~10–25× speedup on Phase 2)

### Why

Phase 2 of both `06_seed_sensitivity.py` and `07_carbon_sweep.py` is a
parallel `ProcessPoolExecutor.map(simulate, args_per_hour)` over the full
dispatch trajectory (~50 k hours × 3–5 strategies per seed/price). At ~5 ms
per `simulate` call serial, that's ~5 minutes per seed/price after
parallelisation across 12 cores.

The per-row args are `(composition, m_dot, T_amb, T_sw, price,
carbon_price)`. Looking at the actual distribution across a 5-year run:

- **Composition** is exact and changes ~once per cargo cycle (12 days). About
  150 unique composition tuples across 5 years.
- **m_dot** is the dispatched flow — quantised to the dispatch flow-level
  grid (15 values) and clustered at the bang-bang bounds {10, 80}.
- **T_amb, T_sw** vary smoothly hour-to-hour; bucketing to 0.5 K loses no
  physical accuracy (CoolProp's HEOS solver is more sensitive than that to
  composition, not to ambient T).
- **price, carbon_price** enter only the cost formula (multiplicative
  outside `simulate`), not the thermo call. Cache the thermo result, apply
  the cost factor per row.

So out of ~50 k rows per strategy, the number of *unique thermo evaluations*
is bounded by `150 compositions × 15 m_dot × 50 T_amb × 50 T_sw ≈ 5 M`
combinatorial, but the realistic on-trajectory count is **~3–8 k unique
points**. A dedupe-before-submit pattern collapses Phase 2 work by **~10–25×**.

### Files

- `src/lng_pinn/plant.py` — add `simulate_thermo_only(comp, m_dot, T_amb,
  T_sw) -> W_total_kwh` helper (or reuse `simulate`, just return the relevant
  scalar) for clarity.
- `scripts/07_carbon_sweep.py` — refactor `_true_cost_for_strategy` and
  `_true_cost_row` to dedupe.
- `scripts/06_seed_sensitivity.py` — same refactor on
  `_eval_true_cost_for_seed_strategy`.

### Design

The cache key bucket function:

```python
def _thermo_key(comp, m_dot, T_amb, T_sw, *, m_dot_bucket=0.5, T_bucket=0.5):
    """Quantised key for thermo memoization.

    Composition is exact (it changes slowly and HEOS is composition-sensitive).
    m_dot and temperatures are bucketed because the dispatch grid is discrete
    and CoolProp is smooth in T.
    """
    return (
        comp,
        round(m_dot / m_dot_bucket) * m_dot_bucket,
        round(T_amb / T_bucket) * T_bucket,
        round(T_sw / T_bucket) * T_bucket,
    )
```

Phase-2 worker becomes thermo-only (no price/carbon arithmetic):

```python
def _simulate_thermo_for_worker(args):
    comp, m_dot, T_amb, T_sw = args
    from lng_pinn.plant import simulate
    try:
        return float(simulate(comp, m_dot, T_amb, T_sw).W_total)
    except ValueError:
        return None
```

Phase-2 driver becomes dedupe → submit unique keys → broadcast:

```python
def _true_cost_for_strategy(dispatch_df, ts_df, carbon_price, ...):
    joined = dispatch_df.join(ts_df, how="inner")
    n = len(joined)

    # 1. Build per-row args and bucketed keys
    rows = [
        (
            tuple(float(getattr(row, c)) for c in COMP_COLS),
            float(row.m_dot), float(row.T_amb), float(row.T_sw),
            float(row.price_eur_mwh),
        )
        for row in joined.itertuples()
    ]
    keys = [_thermo_key(r[0], r[1], r[2], r[3]) for r in rows]

    # 2. Dedupe → run only unique keys through CoolProp
    unique_keys = list(set(keys))
    W_by_key = _run_pool(unique_keys, _simulate_thermo_for_worker)

    # 3. Apply cost formula per row using cached W_total
    out = []
    for i, (key, (_, m_dot, _, _, price)) in enumerate(zip(keys, rows)):
        W = W_by_key.get(key)
        if W is None:
            out.append(np.nan); continue
        elec = price * W * m_dot * 3.6
        carbon = carbon_price * co2_per_kg_fuel(rows[i][0]) * m_dot * 3.6
        out.append(elec + carbon)
    return pd.Series(out, index=joined.index, name="true_cost_eur")
```

### Accuracy contract

The bucketing introduces a deterministic small error:

- 0.5 K T_amb/T_sw bucketing → ≤0.1 K change in ORV outlet enthalpy → ≤0.05 %
  change in W_total (HEOS is smooth in T at fixed composition).
- 0.5 kg/s m_dot bucketing → ≤0.6 % change in pump efficiency at the BEP
  extremes → ≤0.4 % change in W_pump → ≤0.1 % change in W_total
  (W_pump is ~3 % of W_total).

Combined worst case: ≤0.15 % per-row error on W_total → ≤0.15 % per-row cost
error. Since the *saving* we report is +2.50 % across cargo seeds, this is
~6 % of the effect size — well below the seed-noise std (0.79 %) and the
CI half-width (0.49 %). Defensible.

### Acceptance criteria

- On a 100-row test sample, the deduped vs. exact CoolProp cost difference
  is < 0.2 % relative for every row.
- Phase-2 wall-clock on a fresh `carbon_sweep_co2_80.csv` drops from ~5 min
  to ≤30 s on 12 cores (with `--workers 1`).
- Per-price `carbon_sweep.csv` rows agree with the existing 5-year run to
  within 0.05 percentage points on `saving_vs_lagged_pct`.

### Risk

Low. Memoization is a closed-form optimisation; the accuracy contract is
worst-case-bounded analytically. The resume cache schema is unchanged.

---

## E2 — Optional PINN-as-truth Phase 2 with validation sample (~20× on top of E1)

### Why

The v1.4 documentation states the v1.3 PINN matches CoolProp ground truth to
~1e-7 relative error on the LHS test set. That means Phase 2 of the carbon
sweep and seed sensitivity isn't actually *measuring* anything reviewers
would object to — it's redundantly confirming a surrogate that's already
been validated.

The honest move: do Phase 2 CoolProp on a small **validation sample**
(default 5 %) of rows, report the validation error, use the PINN's
own cost prediction for the rest. The PINN cost is in `dispatch_df.cost_eur`
by construction — we just use it directly.

### Files

- `scripts/07_carbon_sweep.py` — add `--validation-sample-frac` flag
  (default 1.0 = current full-CoolProp behaviour).
- `scripts/06_seed_sensitivity.py` — same flag.
- New `results/tables/phase2_validation.csv` — summary of validation-sample
  error (mean rel err, p95 rel err, max rel err) so the paper can cite it.

### Design

```python
def _true_cost_for_strategy(
    dispatch_df, ts_df, carbon_price,
    validation_sample_frac=1.0,
    ...,
):
    joined = dispatch_df.join(ts_df, how="inner")
    n = len(joined)

    if validation_sample_frac >= 1.0:
        # Backward-compat path — full CoolProp re-eval (E1 dedupe still applies)
        return _full_coolprop_eval(joined, carbon_price, ...)

    # Hybrid path
    rng = np.random.default_rng(seed=12345)  # deterministic sample
    n_sample = max(1, int(round(n * validation_sample_frac)))
    sample_idx = np.sort(rng.choice(n, size=n_sample, replace=False))

    # CoolProp truth for the validation sample
    sample_true_costs = _coolprop_eval_subset(joined.iloc[sample_idx], carbon_price, ...)

    # PINN cost for everything (the dispatch cost_eur column already has it,
    # because dispatch was run at this carbon_price)
    out = joined["cost_eur"].to_numpy().copy()
    out[sample_idx] = sample_true_costs.to_numpy()

    # Record validation diagnostics so the paper can quote them
    pinn_at_sample = joined["cost_eur"].iloc[sample_idx].to_numpy()
    rel_err = (sample_true_costs.to_numpy() - pinn_at_sample) / (
        np.abs(pinn_at_sample) + 1e-12
    )
    _append_validation_diagnostics(label=..., rel_err=rel_err)

    return pd.Series(out, index=joined.index, name="true_cost_eur")
```

### Accuracy contract

The hybrid true-cost series is exact on `validation_sample_frac` of rows
and PINN-predicted on the rest. PINN-vs-CoolProp relative cost error is
~1e-7 on the LHS test set, so the substitution error on the unsampled rows
is bounded by that figure.

The validation sample itself is a probe — its rel err distribution is
*reported* in `phase2_validation.csv` and quoted in the paper. If the
sample shows the bound is wider than expected (e.g. p95 > 1e-4) the paper
gets a fact, not an artefact.

### Acceptance criteria

- With `--validation-sample-frac 1.0` (default), bit-for-bit identical to
  pre-v1.5 results.
- With `--validation-sample-frac 0.05`, end-to-end wall-clock on a 10-seed
  sensitivity run drops to ≤45 min on M-series (was ~6 hr at v1.3,
  ~2 hr at v1.5 E1).
- `phase2_validation.csv` contains per-call validation rel-err statistics
  the paper can cite verbatim.

### Risk

Medium. Reviewers may ask "you used the surrogate to validate the
surrogate". The answer is in the validation-sample column: it's a Monte
Carlo probe of the PINN-vs-CoolProp residual, and the residual is below
the noise floor of the saving claim. We anticipate the question and answer
it with a number, not a hand-wave.

---

## E3 — Cross-seed blind-annual cost-table cache (~5 % off Phase 1) — **SCRAPPED**

### Honest re-assessment during implementation

`scripts/06_seed_sensitivity._run_backtest` only runs three strategies
(``aware``, ``lagged``, ``horizon``) — not ``annual`` or ``constant``. So
the annual baseline doesn't appear on the seed-sensitivity hot path at
all; the original v1.5 plan misread that.

In ``07_carbon_sweep`` the annual baseline does run, but the carbon sweep
iterates across **prices** at a fixed seed, and the per-window PINN
forward calls (~5 ms each × 1740 × 9 prices for the annual strategy) sum
to ~80 s — a 1–2 % win on the 7-minute Phase-1 total. Not worth the
implementation complexity.

E3 is dropped from v1.5. The headline speedups remain E1 (dedupe,
~10–25×) and E2 (validation sample, ~20× on top), which already drop
end-to-end wall-clock from ~7 h to ~1 h.

## E3 — (original, kept for context)

### Why

In `06_seed_sensitivity._run_backtest`, the `annual_composition` baseline
fixes composition to the timeseries-wide mean — a value that **does not
depend on the cargo seed**. Currently we recompute its PINN cost table
1740 windows × 10 seeds = 17 400 times when 1740 unique evaluations would
suffice.

This is a smaller win than E1 and E2 (only 1 of 5 strategies, only Phase
1) but it's clean to implement and shaves ~5 minutes off the seed
sensitivity wall-clock.

### Files

- `scripts/06_seed_sensitivity.py` — pre-compute the per-window
  annual-baseline cost tables once before the seed loop, pass into
  `_run_backtest` via a side dict.
- `src/lng_pinn/baseline.py` — accept an optional `precomputed_cost_table`
  kwarg in `optimize_blind_annual` (default None → fall through to the
  per-window computation as today).

### Design

```python
# In 06 main(), before the seed loop:
annual_cost_tables = _precompute_annual_baseline_tables(
    ts, model, scaler, carbon_price=args.carbon_price,
)

# Pass through to _run_backtest:
_run_backtest(ts, model, scaler, seed=seed, ...,
              annual_cost_tables=annual_cost_tables)

# In _run_backtest, when calling the annual baseline:
sched = optimize_blind_annual(
    window, model, scaler, demand_kg, annual_composition, inv["annual"],
    carbon_price_eur_per_t=cp,
    precomputed_cost_table=annual_cost_tables.get(start),
)
```

### Accuracy contract

Bit-identical to the existing per-window computation — just memoized.

### Acceptance criteria

- With and without the cache, `seed_sensitivity_summary.csv` rows agree
  bit-for-bit.
- Seed sensitivity wall-clock drops by ~3–5 minutes.

### Risk

Low. Behaviour-preserving caching.

---

## Expected outcomes after v1.5

| Step | Pre-v1.5 (v1.4) | + E1 (dedupe) | + E2 (5 % validate) | + E3 (annual cache) |
|---|---:|---:|---:|---:|
| Seed sensitivity (10 seeds, 5y) | ~5 hr | ~1.5 hr | ~45 min | ~40 min |
| Carbon sweep (10 prices, 5y) | ~2 hr | ~25 min | ~12 min | ~12 min |
| **End-to-end** | **~7 hr** | **~2 hr** | **~1 hr** | **~50 min** |

Numbers are wall-clock on a 12-core M-series with `--workers 2`. The big
jumps come from E1 (Phase 2 collapses by 10–25×) and E2 (Phase 2 collapses
by another ~20×). E3 is a polish step.

For Germany or any second country, this turns "set it up overnight" into
"set it up at lunch, look at results before dinner."

---

## Implementation order

1. **E1** — biggest, cleanest, no methodology change. ~1 day. Smoke-test
   against a 100-row sample to verify the accuracy contract.
2. **E3** — quick polish. ~2 hours. Optional but landed alongside E1
   because it's so small.
3. **E2** — add the flag but leave default behaviour unchanged. Document
   the `--validation-sample-frac 0.05` workflow as the recommended one
   for second-country runs. ~half a day including the validation CSV.

Each step ships as its own commit with the acceptance criteria reproduced
in the commit message.
