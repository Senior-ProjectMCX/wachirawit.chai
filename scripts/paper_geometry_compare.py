"""
Paper geometry vs the main slab, scored the same way as Compare Method (Simple).

For each MCX photon file, (a, D_B0) of alpha*D_B = a * max(FR_meas, 0) + D_B0 is
re-calibrated with the unchanged method (Nelder-Mead in log10, least squares of g2_sim vs
g2_phys on the closest-row test set, tau every 5th point), then scored on the same 132
trials:
  R^2 per trial (median, p10, n < 0.9), RMSE of g2, BFI-X (Pearson R, median ratio),
  c2/c1 of BFI = c1*FR_s + c2*FR_d + c3 (geometry gate, paper Eq. 4 / 10-12),
  R^2 in the weak corner (shallow <= 5, deep > 20), and per-trial RMSE wins vs main.

  main   data/pathlengths_shallow5.5.npz   120 x 80 x 50 mm slab, bc aaraaa, source x = 40 mm
  paper  data/pathlengths_paper_geom.npz   100 x 50 x 40.5 mm, bc rrrrrr, source x = 35 mm
                                           (src/mcx_engine.py --geometry paper)

Usage: .venv/bin/python scripts/paper_geometry_compare.py
Output: plots/paper_geometry_compare.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
from config import parameters as cfg
import bfi_fit
import data_io
from compare import _synth_curves, _r2, CAL_TAU_STRIDE, BFI_MM2S_TO_FILE

MODELS = {
    "main": os.path.join(ROOT, "data", "pathlengths_shallow5.5.npz"),
    "paper": os.path.join(ROOT, "data", "pathlengths_paper_geom.npz"),
}


def calibrate(det, meta, beta, tau, g2, trials):
    tc = tau[::CAL_TAU_STRIDE]
    ref = g2[:, ::CAL_TAU_STRIDE, :][:, :, trials]

    def loss(p):
        sim = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tc, trials, 10 ** p[0], 10 ** p[1])
        return np.mean((sim - ref) ** 2)

    x0 = [np.log10(cfg.DEFAULT_A_MM2_S_PER_FR), np.log10(cfg.DEFAULT_DB0_MM2_S)]
    r = minimize(loss, x0, method="Nelder-Mead", bounds=[(-9.0, -3.0), (-9.0, -5.0)])
    return 10 ** r.x[0], 10 ** r.x[1], r


def main():
    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    trials = np.sort(pd.read_csv(os.path.join(ROOT, "dataset", "testset_closest.csv"))["Trial_Index"].to_numpy(int))
    ref = g2[:, :, trials]
    fr_s, fr_d = meta["fr_meas_s"][trials], meta["fr_meas_d"][trials]
    corner = (meta["fr_set_s"][trials] <= 5) & (meta["fr_set_d"][trials] > 20)

    bfi_real = bfi_fit.fit_bfi_all(ref, beta, tau) * BFI_MM2S_TO_FILE
    real_c = bfi_fit.regress_c2c1(bfi_real, fr_s, fr_d)
    out = {"n_trials": int(len(trials)), "real_c2c1": [float(c[4]) for c in real_c],
           "paper_c2c1": [cfg.TARGET_C2_C1[s] for s in range(3)], "models": {}}
    print("real c2/c1 (same fitter): " + " / ".join(f"{c[4]:.3f}" for c in real_c))

    rmse_trial = {}
    for name, path in MODELS.items():
        pl = np.load(path)
        rng = np.random.default_rng(42)
        det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
        n_det = [int(len(pl[f"det_{s + 1}_Ls"])) for s in range(3)]
        mean_L = []
        for s in range(3):
            Ls, Ld = pl[f"det_{s + 1}_Ls"].astype(float), pl[f"det_{s + 1}_Ld"].astype(float)
            w = np.exp(-cfg.MU_A_MM * (Ls + Ld))
            mean_L.append([float(np.sum(w * Ls) / w.sum()), float(np.sum(w * Ld) / w.sum())])

        a, db0, res = calibrate(det, meta, beta, tau, g2, trials)
        sim = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)
        bfi_sim = bfi_fit.fit_bfi_all(sim, beta, tau) * BFI_MM2S_TO_FILE
        sim_c = bfi_fit.regress_c2c1(bfi_sim, fr_s, fr_d)

        rows = []
        rmse_trial[name] = np.sqrt(np.mean((sim - ref) ** 2, axis=1))     # (3, n)
        for s in range(3):
            r2 = np.array([_r2(sim[s, :, j], ref[s, :, j]) for j in range(len(trials))])
            rows.append({
                "sds": data_io.SDS_LABELS[s],
                "detected_photons": n_det[s],
                "abs_weighted_mean_L_shallow_deep_mm": mean_L[s],
                "r2_median": float(np.median(r2)),
                "r2_p10": float(np.percentile(r2, 10)),
                "n_r2_below_0.9": int(np.sum(r2 < 0.9)),
                "r2_corner_median": float(np.median(r2[corner])),
                "rmse": float(np.sqrt(np.mean((sim[s] - ref[s]) ** 2))),
                "bfi_X_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_real[s])[0, 1]),
                "bfi_X_median_ratio": float(np.median(bfi_sim[s] / bfi_real[s])),
                "c2c1": float(sim_c[s][4]),
            })
        out["models"][name] = {"photons": os.path.relpath(path, ROOT), "a": a, "db0": db0,
                               "nfev": int(res.nfev), "nit": int(res.nit), "cal_mse": float(res.fun),
                               "per_sds": rows}

        print(f"\n[{name}] a = {a:.4e}, D_B0 = {db0:.4e}  (Nelder-Mead {res.nit} it / {res.nfev} evals, "
              f"cal RMSE {np.sqrt(res.fun):.4f})")
        for r in rows:
            print(f"  {r['sds']}: photons {r['detected_photons']:>7}  meanL s/d {r['abs_weighted_mean_L_shallow_deep_mm'][0]:5.1f}/"
                  f"{r['abs_weighted_mean_L_shallow_deep_mm'][1]:5.1f} mm  R2 med {r['r2_median']:.3f} p10 {r['r2_p10']:.3f} "
                  f"n<0.9 {r['n_r2_below_0.9']:>3}  corner {r['r2_corner_median']:.3f}  RMSE {r['rmse']:.4f}  "
                  f"BFI-X R {r['bfi_X_pearson_r']:.3f} ratio {r['bfi_X_median_ratio']:.2f}  c2/c1 {r['c2c1']:.3f}")

    wins = [[int(np.sum(rmse_trial["paper"][s] < rmse_trial["main"][s])),
             int(np.sum(rmse_trial["main"][s] < rmse_trial["paper"][s]))] for s in range(3)]
    out["paper_vs_main_rmse_wins"] = wins
    print("\nper-trial RMSE wins, paper - main: " + "  ".join(f"{data_io.SDS_LABELS[s]} {w[0]}-{w[1]}"
                                                            for s, w in enumerate(wins)))
    path = os.path.join(ROOT, "plots", "paper_geometry_compare.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
