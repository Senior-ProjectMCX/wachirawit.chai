# wachirawit.chai — DCS two-layer flow phantom simulation

Monte Carlo (MCX) simulation of diffuse correlation spectroscopy g2(τ) curves
for the two-layer flow phantom of Nakabayashi et al. (2023), *Biomed. Opt.
Express* 14(10) 5358, validated against the paper's physical lab data.

## Setup (Apple Silicon)

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

The script builds `pmcxcl` from source because every PyPI wheel, although
tagged `universal2`, contains only an x86_64 binary.

## Pipeline

```bash
python src/main.py run-mcx --nphoton 2e8        # once: photon paths -> data/pathlengths.npz
python src/main.py generate-dataset             # g2 for every flow condition
python src/main.py validate-geometry --synthetic data/Dataset_simulated.csv
python src/main.py compare                      # calibrate a, D_B0; held-out R2/MAE/RMSE/BFI
```

| Module | Role |
| --- | --- |
| `config/parameters.py` | all physical constants and geometry |
| `src/mcx_engine.py` | voxel phantom + MCX run |
| `src/dcs_physics.py` | flow → αD_B → g1/g2 (Wu et al. 2021 Eq. 5, Siegert) |
| `src/bfi_fit.py` | semi-infinite BFI fit, c2/c1 geometry gate |
| `src/data_io.py` | flow table (`dataset/`) and physical reference (`../core/data/`) |
| `src/dataset_generator.py` | synthetic g2 for all trials |
| `src/compare.py` | calibration and held-out metrics |

Physics is driven by the **measured** outlet flow; trials with deep preset
flow = 0 are excluded by default (`--include-deep-zero` to keep them).
The physical reference data is read from `../core/data/`.
