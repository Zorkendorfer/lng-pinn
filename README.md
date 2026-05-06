# LNG-PINN: Composition-aware PINN surrogate for FSRU regasification dispatch

> We present a physics-informed neural network surrogate of FSRU regasification that treats LNG
> composition as a dynamic state, and couple it to day-ahead electricity price signals to compute
> economically optimal send-out schedules. Using the Independence FSRU in Klaipėda as a reference
> and Lithuanian (LT) ENTSO-E day-ahead prices, we show that composition-aware dispatch yields X%
> cost reduction over composition-blind baselines across a multi-year backtest, and that the PINN
> surrogate matches a CoolProp reference simulator to within Y% on energy balance residuals while
> running ~Z× faster.

## Install

```bash
pip install uv
uv sync
```

Copy `.env.example` to `.env` and fill in your ENTSO-E API token.

## Reproduce

```bash
python scripts/01_pull_entsoe.py
python scripts/02_build_dataset.py
python scripts/03_train_pinn.py
python scripts/04_run_dispatch.py
python scripts/05_make_figures.py
```

All figures land in `results/figures/`. All tables land in `results/tables/`.

## Tests

```bash
pytest
```

## Reference plant

Independence FSRU, Klaipėda, Lithuania (55.71°N, 21.13°E).

## License

MIT
