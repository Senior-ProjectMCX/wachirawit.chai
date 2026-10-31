# wachirawit.chai — Model A: DCS two-layer flow phantom simulation

Monte Carlo (MCX) simulation of diffuse correlation spectroscopy g2(τ) curves for the
two-layer flow phantom of Nakabayashi et al. (2023), *Biomed. Opt. Express* 14(10) 5358,
checked against the paper's physical lab data (`../core/data/`).

## Setup (Apple Silicon)

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

The script builds `pmcxcl` from source because every PyPI wheel, although tagged
`universal2`, contains only an x86_64 binary.

## Run

```bash
python src/main.py run-mcx      # once (~45 s): 1e8 photons -> data/pathlengths.npz (already committed)
python src/main.py calibrate    # ~5 min: Nelder-Mead fit of (a, D_B0), metrics, figures -> final_result/
```

## Model

| Step | What | File |
| --- | --- | --- |
| Geometry | box 100 × 50 × 40.5 mm, shallow 5.5 mm + deep 35 mm, all faces Fresnel, pencil source, detectors r = 1 mm at SDS 1/2/3 cm | `config/parameters.py`, `src/mcx_engine.py` |
| Optics | μa 0.056 mm⁻¹, μs′ 0.481 mm⁻¹, g 0.89, n 1.37, 785 nm | `config/parameters.py` |
| Flow → αD_B | αD_B = a · max(FR_meas, 0) + D_B0 (same a for both layers) | `config/parameters.py` |
| g1, g2 | momentum-transfer MC form (Wu et al. 2021 Eq. 5), Siegert g2 = 1 + β g1², β per SDS = mean g2(0) − 1 | `src/dcs_physics.py` |
| Calibration | least squares of g2 (sim vs real), Nelder-Mead in log10 (a, D_B0) | `src/calibrate.py` |
| Metrics | R² per curve (median, mean), pooled RMSE, BFI-X / BFI-Y ratio (semi-infinite fit, paper Eq. 3) | `src/calibrate.py`, `src/bfi_fit.py` |
| Figures | real (black) vs Model A (blue dashed) for every condition | `src/plot_results.py` |

Test set: `dataset/testset_closest_k5.csv` — for each of the 132 flow conditions ((0,0)
excluded) the 5 trials whose measured flow is closest to the preset (660 trials), built by
`scripts/make_testset.py --k 5`.

## Result (`final_result/`)

a = 9.425e-8 mm²/s per ml/min/100g, D_B0 = 6.675e-7 mm²/s (660 trials, 5,000 photons per SDS).

| SDS | 1 cm | 2 cm | 3 cm |
| --- | --- | --- | --- |
| R² median (per trial) | 0.942 | 0.968 | 0.925 |
| R² mean | 0.825 | 0.937 | 0.883 |
| Pooled RMSE of g2 | 0.0332 | 0.0225 | 0.0315 |
| BFI-X ratio median / mean | 0.74 / 0.75 | 1.07 / 1.04 | 1.21 / 1.24 |
| BFI-Y ratio median / mean | 1.41 / 1.43 | 0.94 / 0.92 | 0.62 / 0.64 |

BFI-X = BFI from our semi-infinite fitter on sim vs on real g2; BFI-Y = sim vs the BFI column
of the metadata (lab algorithm). Earlier experiments (geometry variants, Model B comparison,
photon and K sweeps) are in the git history (branch `k5-photon-test`, commit `a7d4638`).
