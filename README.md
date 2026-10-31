# wachirawit.chai — Model A: DCS two-layer flow phantom simulation

Monte Carlo (MCX) simulation of diffuse correlation spectroscopy (DCS) g2(τ) autocorrelation
curves for the two-layer flow phantom of Nakabayashi et al. (2023), *Biomed. Opt. Express*
14(10) 5358, compared against the paper's physical lab measurements.

The workflow has 3 steps:

```
1. run-mcx    simulate photons in the phantom (once)       -> data/pathlengths.npz
2. calibrate  fit a, D_B0 with Nelder-Mead + evaluate       -> final_result/
3. generate   g2 for any flow rate you choose               -> generated/g2_generated.csv
```

**If you only need to generate g2 curves, go straight to step 3.** The outputs of steps 1 and 2
are already in the repo (`data/pathlengths.npz` and `final_result/model_a.json`).

---

## 0. Setup

### Requirements
- Apple Silicon Mac (M1–M4), Homebrew, Python 3.11 (`brew install python@3.11`)
- The physical lab data in `../core/data/` (a folder next to this repo) with `metadata.csv` and
  `g2_SDS1cm.csv`, `g2_SDS2cm.csv`, `g2_SDS3cm.csv`. Only needed for `calibrate`.

```
project/
├── core/data/          physical lab data
└── wachirawit.chai/    this repo
```

### Install the environment

```bash
git clone https://github.com/Senior-ProjectMCX/wachirawit.chai.git
cd wachirawit.chai
bash scripts/setup_env.sh        # creates .venv, installs numpy/scipy/pandas/matplotlib, builds pmcxcl
source .venv/bin/activate
```

`setup_env.sh` builds `pmcxcl` (MCX for OpenCL) from source, because the PyPI wheels contain
only an x86_64 binary and do not run on Apple Silicon. If you only use `generate`, `pmcxcl` is
not needed: `pip install -r requirements.txt` is enough.

Run every command below from the `wachirawit.chai/` folder with `.venv` activated.

---

## 1. `run-mcx`: simulate photons (once)

```bash
python src/main.py run-mcx                     # 1e8 photons, ~45 s on an M4
python src/main.py run-mcx --nphoton 2e8       # change the photon count
```

- Output: `data/pathlengths.npz`, the pathlength (L) and momentum transfer (Y) of every photon
  that reaches a detector, per layer and per SDS (`det_{1,2,3}_{Ls,Ld,Ys,Yd}`).
- This output **does not depend on flow**, so it is reused for every flow rate.
- Re-run only when the geometry or optics in `config/parameters.py` change
  (μa, μs′, g, layer thickness, detector position or radius).

## 2. `calibrate`: fit the flow → αD_B equation and evaluate

```bash
python src/main.py calibrate                   # ~5 min
```

It does three things:
1. Fits `a` and `D_B0` in αD_B = a · max(FR, 0) + D_B0 so that the simulated g2 is as close as
   possible to the measured g2 (least squares, Nelder-Mead), on the 660-trial test set
   (`dataset/testset_closest_k5.csv`).
2. Computes the metrics: R² median/mean, pooled RMSE, BFI-X ratio, BFI-Y ratio (printed).
3. Plots all 132 flow conditions (measured = black solid, Model A = dashed).

Output in `final_result/`:

| File | Content |
| --- | --- |
| `model_a.json` | a, D_B0, β and all metrics (read by `generate`) |
| `conditions/cond_<shallow>_<deep>.png` | one condition, SDS 1/2/3 cm |
| `all_conditions.pdf` | all conditions in one file |
| `grid_SDS{1,2,3}cm.png` | all 132 conditions in one image per SDS |
| `metrics_per_condition.csv` | R² and RMSE per condition |

Options: `--pathlengths <npz file>`, `--testset <csv file>`, `--out <folder>`

## 3. `generate`: g2 for chosen flow rates

```bash
python src/main.py generate --flows 5,20 12,30 0,40           # (shallow, deep) in ml/min/100g
python src/main.py generate --flow-csv my_flows.csv           # read flows from a file
python src/main.py generate --flows 5,20 --out results/a.csv  # choose the output file
```

- `my_flows.csv` must have the columns `FR_shallow, FR_deep`.
- The output (default `generated/g2_generated.csv`) has one row per flow per SDS:

  | FR_shallow | FR_deep | SDS | tau_0us | tau_1us | … | tau_500us |
  | --- | --- | --- | --- | --- | --- | --- |
  | 5 | 20 | 1cm | 1.405 | … | … | … |

- Fast (under 1 s for several flows). No MCX run and no measured data needed.

### From Python

```python
import sys; sys.path.insert(0, "src")
from generate import generate_g2

g2 = generate_g2([(5, 20), (12, 30)])      # shape (n_flows, 3 SDS, 501 tau), tau = 0..500 us
g2_1cm = g2[0, 0]                          # flow (5, 20) at SDS 1 cm

g2 = generate_g2([(5, 20)], a=1e-7, db0=5e-7, beta=0.5)   # override the fitted values
```

---

## Repository layout

| File | Role |
| --- | --- |
| `config/parameters.py` | all constants: optics, geometry, photon count, flow → αD_B equation |
| `src/main.py` | commands `run-mcx`, `calibrate`, `generate`, `all` |
| `src/mcx_engine.py` | builds the voxel phantom and runs MCX |
| `src/dcs_physics.py` | g1/g2 from photons (Wu et al. 2021 Eq. 5 + Siegert) |
| `src/calibrate.py` | Nelder-Mead fit of a, D_B0 and the metrics |
| `src/bfi_fit.py` | semi-infinite BFI fit (paper Eq. 3) for BFI-X/Y |
| `src/plot_results.py` | measured vs Model A figures |
| `src/generate.py` | g2 for chosen flow rates |
| `scripts/make_testset.py` | builds the test set: `--k 5` = the 5 trials per condition whose measured flow is closest to the preset |
| `scripts/setup_env.sh` | environment setup |
| `data/pathlengths.npz` | MCX output, 1e8 photons |
| `dataset/` | flow table (`Dataset_metadata.csv`) and test set |

## Model details

| Part | Value |
| --- | --- |
| Geometry | box 100 × 50 × 40.5 mm: shallow 5.5 mm + deep 35 mm (paper Sec. 2.1), Fresnel on all faces, 0.5 mm voxels |
| Source / detector | pencil beam at x = 35 mm, detectors of radius 1 mm at SDS 1/2/3 cm |
| Optics (785 nm) | μa 0.056 mm⁻¹, μs′ 0.481 mm⁻¹, g 0.89, n 1.37 (both layers) |
| Flow → αD_B | αD_B = a · max(FR, 0) + D_B0, same a for both layers |
| g2 | momentum-transfer g1, g2 = 1 + β·g1², β per SDS = mean measured g2(0) − 1 |

## Current results

a = 9.425e-8 mm²/s per ml/min/100g, D_B0 = 6.675e-7 mm²/s, β = 0.405 / 0.494 / 0.485
(fitted on 660 trials, 5,000 photons per SDS)

| SDS | 1 cm | 2 cm | 3 cm |
| --- | --- | --- | --- |
| R² median (per trial) | 0.942 | 0.968 | 0.925 |
| R² mean | 0.825 | 0.937 | 0.883 |
| Pooled RMSE of g2 | 0.0332 | 0.0225 | 0.0315 |
| BFI-X ratio median / mean | 0.74 / 0.75 | 1.07 / 1.04 | 1.21 / 1.24 |
| BFI-Y ratio median / mean | 1.41 / 1.43 | 0.94 / 0.92 | 0.62 / 0.64 |

- BFI-X = BFI from our fitter (paper Eq. 3), simulated g2 vs measured g2.
- BFI-Y = compared with the BFI column of the metadata (the lab's algorithm, which we do not know).
- Earlier experiments (other geometries, Model B comparison, photon count and K sweeps) are in
  the git history: branch `k5-photon-test`, commit `a7d4638`.
