# LNG-PINN v1.4 — from result to paper

## Why v1.4

v1.3 established the core finding: composition-aware FSRU dispatch beats a
realistic lagged-composition baseline by **+2.25% ± 0.35% (SE), t = 6.5**
across 10 composition seeds at EU-ETS carbon pricing (€80/tCO₂), and the
benefit is carbon-driven (≈0% at €0/tCO₂). v1.4 does not chase a bigger
number — it makes the existing result publication-grade by (a) reporting it
with the correct statistic, (b) contextualising it against a theoretical
ceiling, (c) demonstrating it generalises beyond one price zone, and (d)
explaining the one regime where it fails.

The work is grouped into three tiers. Tier 1 items are things reviewers will
demand; Tier 2 strengthens the core; Tier 3 is explicitly deferred to future
work in the paper. Nothing here touches the PINN surrogate — its ~1e-7
fidelity is settled.

---

## C0 — Statistical framing fix (no compute; do first)

### Why

The headline result is about the **mean saving across cargo-composition
realisations**, whose 2σ confidence interval is ±0.7% (n=10). The current
fig6 overlays the ±2σ **single-seed prediction band** (±2.8%), which answers
a different question ("will one operator on one schedule save?") and makes the
result look statistically marginal when it is in fact decisive (t = 6.5,
p < 1e-4).

### Changes

- `src/lng_pinn/plots.py` — `fig_carbon_sweep` Panel A: draw **two** bands and
  label them unambiguously:
  - CI on the mean (±2·SE ≈ ±0.7%) — the claim the paper makes.
  - Prediction interval (±2σ ≈ ±2.8%) — the single-seed spread, lighter shade.
- Add a small results table `results/tables/seed_significance.csv` with, per
  year and pooled: n, mean, std, SE, t-stat, two-sided p.
- Paper text reports `mean ± SE` and the t-stat, never `mean ± std` alone.

### Acceptance criteria

- `seed_significance.csv` reproduces: 2021 t≈6.6, 2022 t≈1.9, 2023 t≈3.0,
  pooled t≈6.5.
- fig6 Panel A legend distinguishes "95% CI of mean" from "single-seed ±2σ".

### Risk

None — relabelling and a t-test on data already on disk.

---

## Tier 1 — reviewers will demand these

### A — Perfect-foresight upper bound

#### Why

"+2.25%" is more compelling as "+2.25%, capturing X% of the perfect-information
ceiling". A non-causal oracle that sees the entire future composition and price
trajectory bounds the achievable saving; "aware" (which sees only the current
7-day window's composition) should fall below it, and "lagged" below that. The
three-way gap is the figure that tells the whole story.

#### Files

- `src/lng_pinn/baseline.py` — add `optimize_perfect_foresight(...)` that
  optimises over the full horizon with the true composition at every hour and
  no rolling-window restriction (or the longest tractable horizon, documented).
- `scripts/04_run_dispatch.py`, `scripts/07_carbon_sweep.py` — add the oracle
  as a sixth strategy; thread it through the per-strategy cache the same way
  the existing five are handled.
- `src/lng_pinn/plots.py` — new `fig_foresight_gap` (or extend fig6) showing
  constant → lagged → aware → oracle as a savings cascade at €80/tCO₂.

#### Acceptance criteria

- Oracle saving ≥ aware saving ≥ lagged saving at every carbon price (monotone
  information ordering; a violation signals a dispatch bug).
- Report "aware captures {aware/oracle:.0%} of the perfect-foresight ceiling".

#### Risk

Medium. The oracle's horizon choice (full 3-year vs long-but-tractable) must be
documented; MILP size grows with horizon. Mitigate by using the same 7-day
rolling structure but feeding true (not lagged/mean) composition — that's the
honest "perfect composition foresight within the existing planning horizon"
bound, which is the relevant ceiling anyway.

### B — Second price zone (generalisation)

#### Why

One market (Lithuania) invites "does this generalise?". A contrasting zone
shows the mechanism is not LT-specific. Cheapest credibility-per-effort win
because the ENTSO-E pipeline already exists.

#### Files

- `src/lng_pinn/market.py` — already parameterised by `zone`; add the second
  zone code (candidate: Germany `10Y1001A1001A82H` — deep, liquid; or a more
  volatile zone for contrast).
- `scripts/01_pull_entsoe.py` — pull the second zone year-by-year (same 503
  workaround as LT).
- `scripts/02_build_dataset.py` — `build_timeseries(zone=...)` already supports
  it; produce `timeseries_<zone>.parquet`.
- Sweep/seed scripts gain a `--zone` flag; outputs suffixed by zone.
- Paper: side-by-side carbon-sweep panels per zone.

#### Acceptance criteria

- Second-zone carbon sweep completes for {0, 40, 80, 120, 160} at minimum.
- Direction of the effect (positive saving above ~€30/tCO₂) reproduces; the
  magnitude may differ and that difference is itself a discussion point.

#### Risk

Medium. Weather (Open-Meteo) is keyed to the FSRU location, not the price zone
— keep the Klaipėda weather/composition fixed and vary only the price series,
OR state clearly that the second zone is a "price counterfactual" at the same
physical terminal. The latter is cleaner and defensible.

### C — Report significance, not just spread

Covered by C0's `seed_significance.csv`; Tier-1 status because the paper's
quantitative claims must cite it.

---

## Tier 2 — strengthens the core

### D — Explain the 2022 failure

#### Why

2022 (energy crisis) is the one year where saving is marginal (+0.9%, t=1.9).
A reviewer will ask why. Hypothesis: extreme price volatility (2022 mean
≈€230/MWh, σ≈€155 vs ≈€90 normal) makes the optimiser chase price arbitrage,
swamping the carbon-composition signal.

#### Files

- `scripts/05_make_figures.py` / `plots.py` — a diagnostic scatter: per 7-day
  window, x = price volatility, y = realised aware-vs-lagged saving, coloured
  by year. Expect the 2022 cloud at high volatility / low saving.

#### Acceptance criteria

- Negative or flat relationship between window price-volatility and saving,
  with 2022 windows concentrated in the high-volatility / low-saving region.

#### Risk

Low. Diagnostic only; supports a paragraph, doesn't change headline numbers.

### E — Carbon-incidence framing

#### Why

The entire positive result is carbon-driven, so "who pays the CO₂ cost" is the
softest target in the paper. Pre-empt it in Methods.

#### Changes

- Methods paragraph stating the assumed incidence (importer-of-record / full
  pass-through to the regas customer / internalised utility) and a sensitivity
  note that the *relative* saving is invariant to who pays — it depends only on
  the carbon price level, not its allocation.
- Optional: a one-line robustness check that halving the assumed pass-through
  fraction leaves the percentage saving unchanged (it should, since it scales
  the same cost term for all strategies).

#### Risk

Low — mostly writing, one sanity-check run.

### F — Demand-level sensitivity

#### Why

`demand_kg` is fixed; the saving depends on how much flow-shaping slack the
schedule has. A sweep over demand (or tank-turn rate) maps the operating
regime where composition awareness pays.

#### Files

- `scripts/04_run_dispatch.py` — `--demand-factor` flag scaling the existing
  `M_DOT_MAX * 0.6` demand basis.
- A small sweep (e.g. 0.4, 0.6, 0.8 of max) at €80/tCO₂, one figure.

#### Acceptance criteria

- Saving rises as demand leaves more headroom for flow-shaping (monotone or
  single-peaked); documents the regime of applicability.

#### Risk

Low–medium; another sweep dimension but the cache machinery already supports it.

---

## Tier 3 — explicitly future work (state in paper, do not implement)

- **Real cargo-composition data** in place of the synthetic 12-day archetype
  cycle. Highest scientific value, likely infeasible to source pre-submission.
  Name it as the primary future-work item.
- **Plant-parameter sensitivity** (pump η curve, ORV approach ΔT, trim-heater
  efficiency). Mention robustness expectation; defer the sweep.
- **Multi-FSRU / network** dispatch. Out of scope.

---

## Expected paper narrative after v1.4

> Composition-aware FSRU dispatch reduces regasification cost by
> **2.3% ± 0.7%** (95% CI, n=10 composition seeds) versus a realistic
> lagged-composition baseline under EU-ETS carbon pricing, capturing **{X}%**
> of the perfect-foresight ceiling. The benefit is carbon-driven — negligible
> at zero carbon price, rising approximately linearly to a plateau near
> €80–120/tCO₂ — and reproduces across two European price zones. It is robust
> in normal-volatility years (2021: +4.6%, 2023: +1.3%) but is masked by
> price-driven arbitrage in the extreme-volatility 2022 energy crisis
> (+0.9%, n.s.). A physics-by-construction PINN surrogate reproduces the
> CoolProp plant model to ~1e-7 relative cost error, enabling the joint
> composition–dispatch MILP to be solved ~{N}× faster than direct
> equation-of-state evaluation in the loop.

## Implementation order

1. **C0** — statistical framing + fig6 relabel + `seed_significance.csv` (hours).
2. **A** — perfect-foresight oracle + cascade figure (~half day).
3. **B** — second price zone pull + sweep (~half day, mostly compute).
4. **D, E, F** — diagnostics and framing (~1 day total).

Tier 1 (C0 + A + B) is the minimum for a defensible submission. Tier 2 turns
reviewer objections into pre-empted paragraphs. Estimated focused effort: 2–3
days, dominated by compute (zone sweep, oracle sweep) which the resumable
cache machinery already parallelises.
