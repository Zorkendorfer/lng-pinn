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
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and set up
git clone https://github.com/Zorkendorfer/lng-pinn.git
cd lng-pinn
uv sync --extra dev
```

Copy `.env.example` to `.env` and fill in your ENTSO-E API token:

```bash
cp .env.example .env   # then edit .env
```

### macOS (Apple Silicon)

PyTorch will automatically use the **MPS backend** on M-series chips — no extra steps needed.

The dispatch optimiser requires the **CBC solver**. Install it via Homebrew:

```bash
brew install cbc
```

Without CBC, `scripts/04_run_dispatch.py` will fail. (On Linux, install via
`apt install coinor-cbc`; on Windows, CBC is bundled with some Pyomo installations —
or use `pip install cylp`.)

## Reproduce

Run scripts in order from the repo root:

```bash
uv run python scripts/01_pull_entsoe.py          # pull 3 years of LT day-ahead prices
uv run python scripts/02_build_dataset.py        # build training set + timeseries
uv run python scripts/03_train_pinn.py           # train PINN surrogate (~50k steps)
uv run python scripts/04_run_dispatch.py         # run backtest optimisation
uv run python scripts/05_make_figures.py         # generate paper figures
```

All figures land in `results/figures/`. All tables land in `results/tables/`.

Quick sanity check (1000-point dry run, no internet needed after first pull):

```bash
uv run python scripts/02_build_dataset.py --dry-run
```

## Tests

```bash
uv run pytest
```

## Reference plant

Independence FSRU, Klaipėda, Lithuania (55.71°N, 21.13°E).

## License

MIT
