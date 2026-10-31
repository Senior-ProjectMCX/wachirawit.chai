"""
Benchmark of Model A against the physical lab data (formerly compare_simple.py): the minimal
path from simulated g2 to the requested numbers, no train/test split and no extras.

  1. Forward model per test-set trial, driven by the MEASURED flow FR_meas, with the fitted
     (a, D_B0) and beta from final_result/model_a.json (src/main.py calibrate).
  2. g2 metrics vs the physical g2 of the SAME trial, per SDS:
       R^2 = 1 - SS_res/SS_tot per curve (median, mean) and pooled over all curves
       pooled RMSE = sqrt(sum (sim - real)^2 / number of points)
  3. BFI: fit g2(tau) with the semi-infinite model (paper Eq. 3), beta per curve:
       BFI-X = our fitter; ratio BFI-X(sim g2) / BFI-X(physical g2)  (same fitter -> sim error only)
       BFI-Y = BFI column of metadata.csv (lab algorithm, unknown); ratio BFI-X(sim g2) / BFI-Y
     reported as median, mean and Pearson R.
  4. Figures of every condition (plot_results.py).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg
import bfi_fit
import data_io
from dcs_physics import synth_curves


def r2(sim, ref):
    """R^2 = 1 - SS_res/SS_tot over all points of the given arrays."""
    ss_res = np.sum((ref - sim) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def metrics(sim, ref, beta, tau, bfi_y):
    """Per-SDS metrics. sim/ref shape (3, N_TAU, n_trials); bfi_y shape (3, n_trials)."""
    n = ref.shape[2]
    r2_t = np.array([[r2(sim[s, :, k], ref[s, :, k]) for k in range(n)] for s in range(3)])
    bfi_sim = bfi_fit.fit_bfi_all(sim, beta, tau)
    bfi_real = bfi_fit.fit_bfi_all(ref, beta, tau)
    rx, ry = bfi_sim / bfi_real, bfi_sim / bfi_y
    rows = []
    for s in range(3):
        rows.append({
            "sds": data_io.SDS_LABELS[s], "n_trials": int(n),
            "r2_median": float(np.median(r2_t[s])),
            "r2_mean": float(np.mean(r2_t[s])),
            "r2_pooled": float(r2(sim[s], ref[s])),
            "rmse_pooled": float(np.sqrt(np.mean((sim[s] - ref[s]) ** 2))),
            "bfiX_ratio_median": float(np.median(rx[s])),
            "bfiX_ratio_mean": float(np.mean(rx[s])),
            "bfiX_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_real[s])[0, 1]),
            "bfiY_ratio_median": float(np.median(ry[s])),
            "bfiY_ratio_mean": float(np.mean(ry[s])),
            "bfiY_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_y[s])[0, 1]),
        })
    return rows


def run(pathlengths, testset, model_path, out_dir, plot=True):
    from generate import load_model
    from plot_results import plot_all

    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    tau = data_io.TAU_ARRAY
    trials, _ = data_io.load_testset(testset)
    assert not np.any(data_io.excluded_condition(meta["fr_set_s"][trials], meta["fr_set_d"][trials])), \
        "test set contains (0,0) rows"
    a, db0, beta = load_model(model_path)
    det = data_io.load_detectors(pathlengths, cfg.PHOTONS_PER_SDS)
    print(f"benchmark: {len(trials)} trials, a = {a:.4e}, D_B0 = {db0:.4e}, "
          f"beta = {beta[0]:.3f} / {beta[1]:.3f} / {beta[2]:.3f}")

    sim = synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)
    print("fitting BFI (semi-infinite, paper Eq. 3) on sim and physical curves...")
    rows = metrics(sim, g2[:, :, trials], beta, tau, meta["bfi"][:, trials])

    print(f"\n{'SDS':<5}{'R2 med':>8}{'R2 mean':>9}{'R2 pool':>9}{'RMSE':>8}"
          f"{'X med':>7}{'X mean':>8}{'X R':>7}{'Y med':>7}{'Y mean':>8}{'Y R':>7}")
    for r in rows:
        print(f"{r['sds']:<5}{r['r2_median']:>8.3f}{r['r2_mean']:>9.3f}{r['r2_pooled']:>9.3f}"
              f"{r['rmse_pooled']:>8.4f}{r['bfiX_ratio_median']:>7.2f}{r['bfiX_ratio_mean']:>8.2f}"
              f"{r['bfiX_pearson_r']:>7.3f}{r['bfiY_ratio_median']:>7.2f}{r['bfiY_ratio_mean']:>8.2f}"
              f"{r['bfiY_pearson_r']:>7.3f}")
    print("(X = BFI-X ratio sim/physical, same fitter; Y = BFI-Y ratio sim/metadata BFI)")

    os.makedirs(out_dir, exist_ok=True)
    out = {"model": os.path.relpath(model_path, data_io.ROOT),
           "pathlengths": os.path.relpath(pathlengths, data_io.ROOT),
           "testset": os.path.relpath(testset, data_io.ROOT), "per_sds": rows}
    if plot:
        out["n_conditions"] = int(len(plot_all(out_dir, tau, meta, g2, sim, trials)))
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_dir}")
    return rows
