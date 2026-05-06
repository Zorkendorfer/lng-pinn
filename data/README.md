# Data provenance

## data/raw/

All raw files are gitignored and reproduced by `scripts/01_pull_entsoe.py`.

| File pattern | Source | Script |
|---|---|---|
| `da_prices_LT_*.parquet` | ENTSO-E Transparency Platform, Lithuanian bidding zone (10YLT-1001A0008Q) | `01_pull_entsoe.py` |

## data/processed/

All processed files are gitignored and reproduced by `scripts/02_build_dataset.py`.

| File | Schema | Script |
|---|---|---|
| `train.parquet` | CH4, C2H6, C3H8, nC4H10, iC4H10, N2 (mole fractions), m_dot (kg/s), T_amb (K), T_sw (K), W_pump (kWh/kg), W_trim (kWh/kg), W_total (kWh/kg), T_out (K), Q_sw (kWh/kg), exergy_destruction (kWh/kg) | `02_build_dataset.py` |
| `timeseries.parquet` | utc_time index, price_eur_mwh, T_amb, T_sw, CH4, C2H6, C3H8, nC4H10, iC4H10, N2 | `02_build_dataset.py` |

## Notes

- Seawater temperature (T_sw): proxy from Open-Meteo sea surface temperature at 55.71°N, 21.13°E.
- Ambient temperature (T_amb): Open-Meteo ERA5 reanalysis at the same location.
- LNG compositions: synthetic trajectories based on GIIGNL Annual Report 2023 archetypes. See `src/lng_pinn/composition.py`.
