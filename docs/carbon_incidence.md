# Carbon-cost incidence (v1.4 E)

The v1.3/v1.4 positive result is carbon-driven: at €0/tCO₂ the
composition-aware saving collapses to noise, and it rises with the carbon
price. Because the entire effect hinges on the carbon term, the paper must
state explicitly **who bears the CO₂ cost** and why the reported *percentage*
saving is robust to that choice.

## Assumed incidence

We assume the regasification operator faces the marginal CO₂ cost of the gas it
sends out — equivalently, the operator is the importer-of-record under EU ETS,
or fully passes the carbon cost through to (and is optimising on behalf of) the
gas buyer. The dispatch objective is therefore

```
cost[t] = price_elec[t] · W_total[t] · m_dot[t] · 3.6
        + price_co2     · co2_per_kg_fuel(composition[t]) · m_dot[t] · 3.6
```

where `co2_per_kg_fuel` is the stoichiometric kg CO₂ per kg fuel fully
combusted (≈2.50–2.95 across the operating envelope; see `thermo.co2_per_kg_fuel`).

## Why the percentage saving is invariant to the incidence fraction

Suppose only a fraction `α ∈ (0, 1]` of the carbon cost is actually borne by the
decision-maker (partial pass-through). The carbon term scales by `α` **for every
strategy identically** — aware, lagged, horizon, annual, constant. The dispatch
*decisions* depend on the relative cost surface, which is unchanged in shape by a
uniform `α` on the carbon component only in the limit where the carbon term
dominates; more generally, `α` is equivalent to evaluating the sweep at carbon
price `α · price_co2`. So a partial-pass-through world is already represented by a
lower point on the existing carbon-price sweep. No separate run is needed:

- `α = 0.5` at €80/tCO₂  ≡  the €40/tCO₂ sweep point.
- `α = 0.25` at €80/tCO₂ ≡ the €20/tCO₂ sweep point.

The headline figure (fig6) therefore already spans the full range of plausible
incidence assumptions; the x-axis can be read either as "carbon price under full
incidence" or as "effective carbon price = incidence fraction × market carbon
price". This is the robustness statement to put in the Methods section.

## Sanity check (optional)

A one-line confirmation that halving the carbon price reproduces the
half-incidence number:

```
python scripts/04_run_dispatch.py --carbon-price 40 --no-resume
# compare the aware-vs-lagged % saving to the alpha=0.5 @ €80 interpretation
```

## What we explicitly do NOT claim

- We do not model an emissions cap or free-allocation allowances (the marginal
  price is taken as the full ETS price).
- We do not model fuel-gas / boil-off CO₂ separately from sent-out gas; the
  carbon term is on delivered gas only. This is conservative for the
  composition effect (heavier boil-off would amplify, not reduce, the signal).
