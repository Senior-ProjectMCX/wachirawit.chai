"""
Simple compare: the minimal path from simulated g2 to the three requested numbers
(RMSE of g2, R^2 of g2, BFI), with no train/test split and none of the extras
(geometry gate, flow-bin breakdown, subset experiments) of compare.py.

Steps, each tied to Nakabayashi et al. (2023, BOE 14(10) 5358) where the paper defines it:
  1. Forward model (per trial, per SDS), driven by the MEASURED flow FR_meas:
         alpha*D_B = a * max(FR_meas, 0) + D_B0                  (BFI := alpha*D_B, paper Sec. 2.4.1)
         g1(tau)   = momentum-transfer MC form (Wu et al. 2021 Eq. 5, the paper's MC ref [37])
         g2(tau)   = 1 + beta * |g1(tau)|^2                       (paper Eq. 2, Siegert)
     beta per SDS = mean(physical g2(0)) - 1.
  2. Calibrate the two physical constants (a, D_B0) ONCE: least squares of g2_sim vs g2_phys,
     one representative trial per preset combo (all 108 combos, tau every 5th point).
  3. Metrics vs the physical g2 of the SAME trial: RMSE and R^2 of g2 (not defined by the
     paper -- standard statistics).
  4. BFI: fit g2(tau) with the semi-infinite homogeneous model (paper Eq. 3), beta per curve,
     on the sim curve and on the physical curve with the same fitter:
         BFI-A = BFI(sim g2)  vs  BFI(physical g2)               (isolates the sim error)
         BFI-B = BFI(sim g2)  vs  BFI column of metadata.csv      (paper/lab fit; the absolute
                                                                   scale differs from our fitter)
Deep-flow == 0 trials (including (0,0)) are excluded, per Goal.md.

Usage:  .venv/bin/python src/compare_simple.py [--per-combo 5]   (0 = every trial)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg
import bfi_fit
import data_io
from compare import _synth_curves, _r2, CAL_TAU_STRIDE, BFI_MM2S_TO_FILE


def run(pathlengths_path, per_combo, seed, out_json):
    meta = data_io.load_physical_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    tau = data_io.TAU_ARRAY
    fr_s, fr_d = meta["fr_meas_s"], meta["fr_meas_d"]

    keep = np.where(meta["fr_set_d"] != 0.0)[0]
    combos = {}
    for i in keep:
        combos.setdefault((meta["fr_set_s"][i], meta["fr_set_d"][i]), []).append(i)
    rng = np.random.default_rng(seed)
    trials = []
    for idxs in combos.values():
        idxs = np.array(idxs)
        if per_combo and len(idxs) > per_combo:
            idxs = rng.choice(idxs, per_combo, replace=False)
        trials.extend(idxs.tolist())
    trials = np.array(sorted(trials))
    print(f"{len(combos)} preset combos, {len(trials)} trials evaluated "
          f"(Deep_Flow==0 excluded: {meta['n_trials'] - len(keep)} trials)")

    pl = np.load(pathlengths_path)
    drng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, drng) for s in range(3)}

    # --- 2. calibrate (a, D_B0) once, on one representative trial per combo ---
    cal_trials = np.array([min(v) for v in combos.values()])
    tau_cal = tau[::CAL_TAU_STRIDE]
    ref_cal = phys_g2[:, ::CAL_TAU_STRIDE, :][:, :, cal_trials]

    def loss(p):
        sim = _synth_curves(det, fr_s, fr_d, beta, tau_cal, cal_trials, 10.0 ** p[0], 10.0 ** p[1])
        return np.mean((sim - ref_cal) ** 2)

    x0 = [np.log10(cfg.DEFAULT_A_MM2_S_PER_FR), np.log10(cfg.DEFAULT_DB0_MM2_S)]
    res = minimize(loss, x0, method="Nelder-Mead", bounds=[(-9.0, -3.0), (-9.0, -5.0)])
    a, db0 = 10.0 ** res.x[0], 10.0 ** res.x[1]
    print(f"calibrated a={a:.4e} mm^2/s per (ml/min/100g), D_B0={db0:.4e} mm^2/s "
          f"(on {len(cal_trials)} combos, {res.nfev} evaluations)")

    # --- 1. forward model for the evaluated trials, 3-4. metrics ---
    sim = _synth_curves(det, fr_s, fr_d, beta, tau, trials, a, db0)
    ref = phys_g2[:, :, trials]
    print("fitting BFI (semi-infinite, paper Eq. 3) on sim and physical curves...")
    bfi_sim = bfi_fit.fit_bfi_all(sim, beta, tau) * BFI_MM2S_TO_FILE      # 1e-9 cm^2/s
    bfi_phys = bfi_fit.fit_bfi_all(ref, beta, tau) * BFI_MM2S_TO_FILE
    bfi_file = meta["bfi"][:, trials]

    rows = []
    for s in range(3):
        d = sim[s] - ref[s]
        r2t = [_r2(sim[s][:, j], ref[s][:, j]) for j in range(len(trials))]
        rows.append({
            "sds": data_io.SDS_LABELS[s], "n_trials": int(len(trials)),
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "r2_per_trial_median": float(np.median(r2t)),
            "r2_pooled": float(_r2(sim[s], ref[s])),
            "bfi_A_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_phys[s])[0, 1]),
            "bfi_A_median_ratio": float(np.median(bfi_sim[s] / bfi_phys[s])),
            "bfi_A_rmse": float(np.sqrt(np.mean((bfi_sim[s] - bfi_phys[s]) ** 2))),
            "bfi_B_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_file[s])[0, 1]),
            "bfi_B_median_ratio": float(np.median(bfi_sim[s] / bfi_file[s])),
            "our_fitter_vs_file_ratio": float(np.median(bfi_phys[s] / bfi_file[s])),
        })

    print("\n=========== Simple compare (sim vs physical, same trials) ===========")
    print(f"{'SDS':<5}{'RMSE':>9}{'R2/trial':>10}{'R2 pooled':>11}{'BFI-A R':>9}{'A ratio':>9}"
          f"{'BFI-B R':>9}{'B ratio':>9}")
    for r in rows:
        print(f"{r['sds']:<5}{r['rmse']:>9.4f}{r['r2_per_trial_median']:>10.3f}{r['r2_pooled']:>11.3f}"
              f"{r['bfi_A_pearson_r']:>9.3f}{r['bfi_A_median_ratio']:>9.2f}"
              f"{r['bfi_B_pearson_r']:>9.3f}{r['bfi_B_median_ratio']:>9.2f}")
    print("(R2 target ~0.95; BFI-A: sim vs physical g2 same fitter; BFI-B: vs metadata.csv BFI column)")
    print("our fitter on physical g2 / file BFI: "
          + "  ".join(f"{r['sds']}={r['our_fitter_vs_file_ratio']:.2f}" for r in rows)
          + "   (!=1 => BFI-B absolute scale not comparable)")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"a_mm2_s_per_fr": a, "db0_mm2_s": db0, "n_combos": len(combos),
                   "n_trials": int(len(trials)), "per_sds": rows}, f, indent=2)
    print(f"saved {out_json}")
    return rows


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="Simple sim-vs-physical comparison (RMSE, R2, BFI)")
    ap.add_argument("--pathlengths", default=os.path.join(root, "data", "pathlengths.npz"))
    ap.add_argument("--per-combo", type=int, default=5, help="trials per preset combo; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(root, "plots", "metrics_simple.json"))
    args = ap.parse_args()
    run(args.pathlengths, args.per_combo, args.seed, args.out)


if __name__ == "__main__":
    main()
