# LNG-PINN: Project Execution Plan (v1 → preprint)

A 10-week plan to ship a single dated, publishable artifact: a physics-informed surrogate of FSRU regasification coupled to ENTSO-E day-ahead prices, with LNG composition as a dynamic state variable. Independence FSRU (Klaipėda) is the reference plant.

The whole point is to **finish v1**, not to maximize scope. Read Section 2 (scope cuts) twice.

---

## 1. Working title and one-paragraph claim

**Working title:** *Composition-aware physics-informed surrogate for FSRU regasification dispatch under day-ahead electricity prices.*

**One-paragraph claim (write this BEFORE writing code, refine weekly):**
> We present a physics-informed neural network surrogate of FSRU regasification that treats LNG composition as a dynamic state, and couple it to day-ahead electricity price signals to compute economically optimal send-out schedules. Using the Independence FSRU in Klaipėda as a reference and Lithuanian (LT) ENTSO-E day-ahead prices, we show that composition-aware dispatch yields X% cost reduction over composition-blind baselines across a multi-year backtest, and that the PINN surrogate matches a CoolProp reference simulator to within Y% on energy balance residuals while running ~Z× faster.

If you can't write a clean version of this paragraph by end of week 2, scope is wrong.

---

## 2. Scope: what's in v1, what is OUT

**IN (do these, finish them):**
- Steady-state regasification model in CoolProp as ground-truth simulator
- ENTSO-E LT day-ahead price ingestion (3+ years history)
- Synthetic but realistic LNG cargo composition trajectories
- PINN surrogate for energy consumption + outlet state given (composition, send-out flow, ambient T, seawater T)
- Offline economic dispatch optimization (no real-time control)
- One baseline (composition-blind dispatch) for comparison
- Reproducibility artifact: GitHub repo with frozen env, README, data-pull script, one figure-generating script
- arXiv preprint or equivalent dated writeup

**OUT (you will be tempted, refuse):**
- Dynamic/transient regasification model (steady-state per dispatch interval is enough)
- Multi-FSRU coordination
- Real plant data (use public Independence specs only; don't chase proprietary data)
- Stochastic optimization / scenario trees
- Reinforcement learning policy
- Comparison to >1 baseline
- Hyperparameter sweeps beyond a small grid
- Frontend / dashboard / Streamlit
- Extension to LNG bunkering, cold energy recovery, or hydrogen — these are v2+
- Rewriting any part in C++ or Fortran for "performance"

If a teammate, advisor, or reviewer suggests adding any of the OUT list, say "v2." Write it on a sticky note above your monitor.

---

## 3. Repository structure (create on day 1, keep stable)

```
lng-pinn/
├── README.md                  # how to run, results, citation
├── pyproject.toml             # pinned deps, project metadata
├── .python-version            # 3.11
├── .gitignore
├── LICENSE                    # MIT or Apache-2.0
├── data/
│   ├── raw/                   # gitignored; pulled by scripts
│   ├── processed/             # gitignored; built by scripts
│   └── README.md              # what each file is, how it was made
├── src/lng_pinn/
│   ├── __init__.py
│   ├── thermo.py              # CoolProp wrappers, mixture handling
│   ├── plant.py               # FSRU steady-state model (ground truth)
│   ├── market.py              # ENTSO-E ingestion
│   ├── composition.py         # cargo composition trajectories
│   ├── dataset.py             # builds (X, y) for PINN training
│   ├── pinn.py                # network, residuals, training loop
│   ├── dispatch.py            # economic optimizer
│   ├── baseline.py            # composition-blind baseline
│   └── plots.py               # all figures generated here
├── scripts/
│   ├── 01_pull_entsoe.py
│   ├── 02_build_dataset.py
│   ├── 03_train_pinn.py
│   ├── 04_run_dispatch.py
│   └── 05_make_figures.py
├── tests/
│   ├── test_thermo.py
│   ├── test_plant.py
│   └── test_pinn_residuals.py
├── notebooks/
│   └── exploration/           # gitignored except a final results.ipynb
├── paper/
│   ├── main.tex
│   ├── refs.bib
│   └── figures/
└── results/
    ├── models/                # trained checkpoints
    ├── figures/
    └── tables/
```

Every script runs from repo root with `python scripts/0X_*.py` and produces a file. No hidden state between scripts. Each script logs its config and git SHA into the output.

---

## 4. Phase plan with binary checkpoints

Each phase has a **binary done condition.** If you can't tick the box, you're not done — don't move on.

### Phase 0 — Setup (Week 1, ~5–8 hours)

**Tasks:**
1. Register for ENTSO-E Transparency Platform account, generate API security token. (Email request, takes 24–48h. Do this **first**, before anything else.)
2. Create GitHub repo `lng-pinn`, MIT or Apache-2.0 license.
3. `uv` or `pdm` for env management on Apple Silicon. Pin Python 3.11 (PyTorch MPS most stable here).
4. Pinned deps in `pyproject.toml`:
   - `numpy`, `scipy`, `pandas`, `pyarrow`
   - `coolprop` (the `CoolProp` PyPI package)
   - `entsoe-py`
   - `torch` (with MPS support)
   - `pyomo` + `ipopt` (via `idaes-pse` or system install) — or `cvxpy` for a simpler LP/QP path
   - `matplotlib`, `seaborn`
   - dev: `pytest`, `ruff`, `mypy`, `pre-commit`
5. Set up pre-commit with ruff + mypy.
6. Write **one paragraph** in `README.md` describing the project goal. This is the public version of Section 1.
7. **Time-boxed literature scan: 4 hours, hard cap.** Search terms: "PINN regasification", "LNG terminal optimization", "FSRU dispatch", "physics-informed surrogate process". Record 8–15 papers in `paper/refs.bib` with one-line notes. Stop at 4 hours regardless of completeness — you'll add more during writeup.

**Done when:** repo public, CI passing on an empty test, ENTSO-E token in `.env.example` (token itself in untracked `.env`), one-paragraph project description in README.

---

### Phase 1 — CoolProp ground-truth plant model (Weeks 2–3, ~15–20 hours)

This is the **reference simulator** the PINN will be trained against. Steady-state, single regasification train.

**Physical model:**
- Inputs: LNG composition vector (mole fractions of CH₄, C₂H₆, C₃H₈, n-C₄H₁₀, i-C₄H₁₀, N₂); send-out mass flow rate (kg/s); ambient air T (K); seawater T (K); LNG inlet T and P (typically ~110 K, ~1 bar after pump-up: ~80 bar).
- Equipment: cryogenic pump (LNG to high pressure), open/intermediate-rack vaporizer (seawater or glycol-water heated), trim heater. Independence FSRU uses propane intermediate fluid in some configurations — pick **one** equipment topology and document it.
- Outputs: send-out gas T and P; total electrical energy consumption (kWh per kg gas sent out); seawater duty; exergy destruction.

**Implementation:**
- `src/lng_pinn/thermo.py`: thin wrappers around `CoolProp.AbstractState("HEOS", "Methane&Ethane&...")`. Use HEOS backend (open-source, mixture-capable). Document its known limitations (poor accuracy for some heavy hydrocarbons near critical) in module docstring.
- `src/lng_pinn/plant.py`: function `simulate(composition, m_dot, T_amb, T_sw, P_out=80e5) -> PlantOutput` returning a dataclass with all outputs above. Pure function, no class state.
- Validation:
  - Energy balance closure to <0.5% across a sweep of inputs.
  - Pump work matches `(P_out - P_in) * v_lng / η_pump` to within 5%.
  - Compare lower heating value of outgoing gas to published Wobbe/calorific spec for a typical Qatari or US Gulf cargo composition.
- Tests in `tests/test_plant.py`: at least 5 reference points covering corners of input space.

**Done when:** `python scripts/02_build_dataset.py --dry-run` runs the simulator over a 1000-point Latin hypercube of inputs in <60 seconds, all energy balances close to <0.5%, all tests pass.

**Pitfall:** CoolProp HEOS mixture initialization is slow; cache `AbstractState` objects by composition hash. Don't re-initialize per call.

---

### Phase 2 — Market and composition data pipeline (Week 3–4, ~10 hours)

**Tasks:**
1. `src/lng_pinn/market.py`: `pull_da_prices(start, end, zone="LT") -> pd.DataFrame`. Use `entsoe-py`'s `EntsoePandasClient.query_day_ahead_prices`. Cache to `data/raw/da_prices_LT_<start>_<end>.parquet`.
2. Pull 3+ years of LT day-ahead prices. Inspect for gaps. Document any imputation in code.
3. `src/lng_pinn/composition.py`: synthetic cargo trajectories.
   - Pick 4–6 archetypal compositions (US Gulf, Qatar, Norway, Algeria, US East Coast). Each has a representative mole-fraction vector — find these in published GIIGNL or IGU reports. Cite the source.
   - Build a "cargo schedule": every ~10–14 days, a new cargo arrives, composition switches stepwise. Tank mixing is approximated as a 5-day linear blend between old and new compositions.
   - Output: hourly composition time series aligned to the price index.
4. Ambient/seawater T: pull from a public source. Options in order of preference: ECMWF ERA5 reanalysis (free, requires CDS account), Copernicus Marine Service for seawater, or Open-Meteo as a quick proxy. Klaipėda port lat/lon: ~55.71°N, 21.13°E.
5. `src/lng_pinn/dataset.py`: `build_training_set(N=20000) -> (X, y)` where X is sampled across realistic operating envelope (composition mix + flow + temps) and y is the CoolProp simulator output. Save to `data/processed/train.parquet`.

**Done when:** `data/processed/train.parquet` exists with N≥20000 samples, schema documented in `data/README.md`, hourly time-aligned dataframe of (price, composition vector, T_amb, T_sw) for 3+ years saved to `data/processed/timeseries.parquet`.

**Pitfall:** ENTSO-E rate limits and silent gaps. Always check `df.isna().sum()` and the index for missing hours after pulling.

---

### Phase 3 — PINN surrogate (Weeks 5–6, ~25–30 hours)

**Architecture:**
- Pure PyTorch (not DeepXDE — you want full control for the writeup, and DeepXDE adds a dependency you'll regret).
- MLP: input dim = 6 (composition) + 3 (m_dot, T_amb, T_sw) = 9. Output dim = 4 (W_pump, W_total, T_out, exergy_destruction).
- 5 hidden layers, 128 units, `tanh` activation (smooth gradients for residuals). Sinusoidal or RFF input embedding optional — only add if plain MLP underfits.
- Normalize inputs and outputs to unit variance. Save scaler to checkpoint.
- Apple Silicon: `device = "mps"` if available, else `cpu`. MPS works fine for this size; don't bother with CUDA.

**Loss:**
$$\mathcal{L} = \lambda_d \cdot \mathcal{L}_{data} + \lambda_e \cdot \mathcal{L}_{energy} + \lambda_m \cdot \mathcal{L}_{mass}$$

- `L_data`: MSE between NN output and CoolProp simulator output on the training set.
- `L_energy`: residual of the steady-state energy balance (input enthalpy + work in = output enthalpy + heat out), evaluated on a separate set of *collocation* points sampled from the same input distribution but **without** querying the simulator.
- `L_mass`: trivial in steady-state but include for symmetry; mostly a sanity check.
- Loss balancing: start with all λ = 1, then use the **NTK-based or gradient-norm balancing scheme** (Wang et al. 2021 / 2022) updated every 1000 steps. Document the exact rule used.

**Training:**
- Adam, lr 1e-3, cosine decay to 1e-5, ~50k steps.
- Batch 512 data + 512 collocation points.
- Validation split 10%, early stop on val loss.
- Log to `tensorboard` or `wandb` (offline mode if you don't want an account).
- Save best checkpoint to `results/models/pinn_v1.pt`.

**Evaluation:**
- On held-out test set: relative error per output. Target: <2% MAE on W_total, <0.5 K on T_out.
- Compare wall-clock: 1000 PINN inferences vs 1000 CoolProp calls. Report ratio.
- Sanity: predict on inputs outside training range (compositional extrapolation) and document failure mode honestly.

**Done when:** trained model passes evaluation thresholds, `tests/test_pinn_residuals.py` confirms energy balance residual <1% on 100 random samples, checkpoint and scaler saved, training command reproducible from `scripts/03_train_pinn.py`.

**Pitfall:** PINN loss balancing is the #1 way these projects fail. If physics loss is 1000× the data loss, the model just fits physics and ignores data, or vice versa. Spend a day on balancing if needed; don't ship an unbalanced model.

---

### Phase 4 — Economic dispatch (Weeks 6–7, ~10–15 hours)

**Problem:**
Given a horizon (e.g., 7 days, hourly resolution), choose send-out flow $\dot{m}_t$ at each hour to minimize total electricity cost subject to:
- Cumulative send-out over the horizon ≥ contracted demand
- Per-hour flow limits (min turndown, max throughput)
- Inventory constraint (LNG tank level stays in bounds)
- Composition is exogenous (from cargo schedule)

Cost at hour $t$: $\text{price}_t \times W_{\text{total}}(\dot{m}_t, x_t, T_{amb,t}, T_{sw,t})$, where $W_{\text{total}}$ is the PINN.

**Implementation:**
- Two paths, pick one:
  - **Path A (recommended for v1):** Discretize $\dot{m}$ into 10–20 levels per hour, solve as MILP via Pyomo + CBC (free) or Gurobi (free academic). PINN evaluated at each grid point in preprocessing → lookup table per hour. Clean, fast, defensible.
  - **Path B:** Treat as continuous NLP, embed PINN as differentiable constraint via `torch` autodiff into IPOPT through `pyomo.environ.ExternalFunction`. More elegant, more pain. Save for v2.
- `src/lng_pinn/dispatch.py`: function `optimize(horizon_df, pinn, demand, inv0) -> Schedule`.
- `src/lng_pinn/baseline.py`: composition-blind baseline that uses an averaged composition (annual mean) for the same optimization. Same demand, same inventory, same prices.

**Done when:** Both optimizers run end-to-end on a 1-month rolling horizon backtest across the full 3-year dataset, schedules and costs saved to `results/tables/dispatch_v1.parquet`, baseline costs saved to `results/tables/baseline_v1.parquet`.

---

### Phase 5 — Composition sensitivity study (Week 7–8, ~10 hours)

This is the **novel result.** Don't skimp.

**Analyses:**
1. Cost delta: composition-aware vs blind, broken down by month, season, year. Bar chart + table.
2. Sensitivity: synthetically vary cargo schedule (more vs less compositional variability). Plot cost savings vs variability metric.
3. Operational difference: when does composition-aware dispatch shift load? Heatmap of $(\text{price quantile}, \text{methane number})$ → load shift.
4. Surrogate fidelity check: re-run optimal schedules through the CoolProp ground truth; report how often the PINN-predicted cost differs from the true cost by >1%.

**Done when:** 4 figures exist in `results/figures/`, each generated by a single function in `src/lng_pinn/plots.py`, all tables exported to CSV.

---

### Phase 6 — Writeup and ship (Weeks 8–10, ~25 hours)

**Paper structure (target ~6–8 pages, single column or arXiv-standard):**
1. Abstract (write last, 150 words)
2. Introduction & related work (~1 page; lean on the lit-scan from Phase 0)
3. System and problem statement (FSRU diagram, decision variables, ENTSO-E framing)
4. Methods: ground-truth simulator, PINN architecture and loss, dispatch formulation
5. Data: ENTSO-E pull, cargo composition modeling, weather data
6. Results: PINN fidelity + speedup, dispatch backtest, composition sensitivity
7. Discussion: limitations (steady-state, synthetic compositions, single FSRU), what would change with real plant data
8. Reproducibility statement: link to repo, exact commit hash, env file
9. References

**Logistics:**
- Use the arXiv-friendly LaTeX template (`\documentclass{article}` with `geometry`, `siunitx`, `booktabs`, `graphicx`, `hyperref`, `cleveref`).
- Push to arXiv as `cs.LG` cross-listed to `eess.SY` and `physics.ao-ph` (or wherever feels right; arXiv moderators may reclassify).
- Tag a GitHub release `v1.0.0` matching the arXiv version.
- DOI via Zenodo by linking the GitHub release.
- Tweet/post once, link to arXiv + repo. Do not edit. Move on to v2.

**Done when:** arXiv preprint posted, Zenodo DOI minted, GitHub release tagged, README links to both.

---

## 5. Tech stack: specific choices and rationale

| Choice | Rationale |
|---|---|
| Python 3.11 | PyTorch MPS most stable; CoolProp wheels available |
| `uv` | Fast, deterministic, modern; works on Apple Silicon |
| PyTorch over JAX | MPS support is rougher in JAX; you want to spend brain on the science, not on Metal kernel debugging |
| CoolProp HEOS over REFPROP | Free, mixture-capable, reproducible. REFPROP would be more accurate but locks reproducibility behind a license |
| Pyomo + CBC over `cvxpy` | MILP support is cleaner in Pyomo; CBC is free |
| `entsoe-py` over raw API | Maintained, handles auth, returns pandas |
| MLP over Fourier features / SIREN | Simplest thing that works; document if you needed to upgrade |
| Pure PyTorch over DeepXDE | Full control of loss balancing, fewer deps, easier to publish |

---

## 6. Reproducibility checklist (every box ticked before you call it done)

- [ ] `pyproject.toml` with all deps pinned to exact versions
- [ ] `.python-version` pinned
- [ ] `README.md` with: project description, install steps, exact commands to reproduce every figure
- [ ] All scripts log git SHA and config to output
- [ ] `data/raw/` files have a one-line provenance comment in `data/README.md`
- [ ] Trained model checkpoint committed via Git LFS or hosted on Zenodo with link in README
- [ ] One end-to-end test: `make reproduce` (or shell script) runs phases 2–5 from scratch on a small sample and produces a sentinel figure
- [ ] arXiv version, GitHub tag, and Zenodo DOI all reference the same commit SHA

---

## 7. Anti-scope list (read weekly)

You will be tempted to:
- Add a transient regasification model "for realism" → **NO.** Steady-state per dispatch interval.
- Switch to JAX "for speed" → **NO.** Speed is not the bottleneck.
- Add a Streamlit dashboard "for the demo" → **NO.** A static figure is enough.
- Compare to 4 baselines → **NO.** One is enough for v1.
- Wait for "real" cargo composition data → **NO.** Synthetic but cited is fine for v1.
- Rewrite the optimizer in Rust → **NO.**
- Add an LLM-based scenario generator → **NO.** What.
- Generalize to "any LNG terminal" → **NO.** Independence FSRU. One plant. v1.

If you violate this list once, that's human. Twice, you're stalling. Three times, the project is dead — close the laptop and ask why.

---

## 8. Definition of done

v1 is done when:
1. arXiv preprint is posted with a permanent ID.
2. GitHub repo is public, tagged `v1.0.0`, with a Zenodo DOI.
3. A stranger can clone the repo, run `make reproduce`, and get the headline figure within 30 minutes on a Mac M-series.
4. The headline number ("X% cost reduction from composition-aware dispatch") appears in the abstract and is reproducible to within Monte Carlo noise.
5. You have *not* started v2 yet. Sit with v1 being public for at least one week before opening the next file.

---

## 9. Day 1 (today) checklist

If you do nothing else after reading this, do these three things in order:

1. **Email ENTSO-E** for an API token. They take 1–2 days; your project is blocked until you have one.
2. **Create the GitHub repo** with the structure in Section 3. Empty files are fine. Push.
3. **Write the one-paragraph claim** from Section 1 in `README.md`. Even a bad version. Update it weekly.

Everything else can start tomorrow.
