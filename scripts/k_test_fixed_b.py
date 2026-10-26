"""
Model A vs Model B (original) vs Model B (pathlength-unit bug fixed) on the K-closest test set.

  Test set   dataset/testset_closest_k{K}.csv (scripts/make_testset.py --k K): the K trials per
             condition whose measured flow is closest to the preset, (0,0) excluded.
  Model A    paper-geometry photons, alpha*D_B = a*max(FR_meas, 0) + D_B0, beta per SDS.
             (a, D_B0) re-fitted on this test set with the unchanged method (Nelder-Mead, log10,
             least squares of g2, tau every 5th point). Held-out check: fit on ranks 1..K-1,
             score on rank K (one unseen trial per condition).
  Model B    Wichuda_r as her code runs it (../wichuda_g2): her photons, preset flow,
             alpha*D_B = 0.4e-6*FR, beta per condition.
  Model B fixed   the same, but her photon pathlengths are multiplied by unitinmm = 0.5 (MCX
             returns them in voxel units; her code uses them as mm). Done on a copy of the
             loaded array -- Wichuda_r itself is not modified.

Usage: .venv/bin/python scripts/k_test_fixed_b.py [--k 5]
Output: plots/k{K}_test.json, plots/k{K}_test_per_curve.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "wichuda_g2"))
from config import parameters as cfg
import bfi_fit
import data_io
from compare import _synth_curves, _r2, CAL_TAU_STRIDE, BFI_MM2S_TO_FILE
import wichuda_g2 as wg

B_SCALE = 0.4e-6          # Wichuda_r commit 64985e7
VOXEL_MM = 0.5            # her unitinmm
LAB = ["1cm", "2cm", "3cm"]


def calibrate(det, meta, beta, tau, g2, trials):
    tc = tau[::CAL_TAU_STRIDE]
    ref = g2[:, ::CAL_TAU_STRIDE, :][:, :, trials]

    def loss(p):
        sim = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tc, trials, 10 ** p[0], 10 ** p[1])
        return np.mean((sim - ref) ** 2)

    x0 = [np.log10(cfg.DEFAULT_A_MM2_S_PER_FR), np.log10(cfg.DEFAULT_DB0_MM2_S)]
    r = minimize(loss, x0, method="Nelder-Mead", bounds=[(-9.0, -3.0), (-9.0, -5.0)])
    return 10 ** r.x[0], 10 ** r.x[1], r


def run_b(detp, meta, g2, trials, tau):
    wg.flow_to_alpha_db = lambda f: max(float(f), 0.0) * B_SCALE
    out = np.zeros((3, len(tau), len(trials)))
    cache = {}
    for k, i in enumerate(trials):
        key = (meta["fr_set_s"][i], meta["fr_set_d"][i])
        if key not in cache:
            same = np.where((meta["fr_set_s"] == key[0]) & (meta["fr_set_d"] == key[1]))[0]
            b = [wg.beta_from_physical_trials(g2[s][:, same].T) for s in range(3)]
            cache[key] = wg.generate_g2(key[0], key[1], b, detp=detp, tau_seconds=tau)
        out[:, :, k] = cache[key]
    return out


def score(sim, ref, beta, tau, bfi_real, bfi_y):
    n = ref.shape[2]
    bfi = bfi_fit.fit_bfi_all(sim, beta, tau) * BFI_MM2S_TO_FILE
    r2 = np.array([[_r2(sim[s, :, k], ref[s, :, k]) for k in range(n)] for s in range(3)])
    rx, ry = bfi / bfi_real, bfi / bfi_y
    summ = {
        "r2_median": [float(np.median(r2[s])) for s in range(3)],
        "r2_mean": [float(np.mean(r2[s])) for s in range(3)],
        "r2_p10": [float(np.percentile(r2[s], 10)) for s in range(3)],
        "rmse_pooled": [float(np.sqrt(np.mean((sim[s] - ref[s]) ** 2))) for s in range(3)],
        "bfiX_ratio_median": [float(np.median(rx[s])) for s in range(3)],
        "bfiX_ratio_mean": [float(np.mean(rx[s])) for s in range(3)],
        "bfiX_fold_error": [float(np.exp(np.median(np.abs(np.log(rx[s]))))) for s in range(3)],
        "bfiY_ratio_median": [float(np.median(ry[s])) for s in range(3)],
        "bfiY_ratio_mean": [float(np.mean(ry[s])) for s in range(3)],
    }
    return summ, r2, rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    ts = pd.read_csv(os.path.join(ROOT, "dataset", f"testset_closest_k{args.k}.csv"))
    ts = ts.sort_values("Trial_Index").reset_index(drop=True)
    trials = ts["Trial_Index"].to_numpy(int)
    assert not np.any(data_io.excluded_condition(meta["fr_set_s"][trials], meta["fr_set_d"][trials]))
    ref = g2[:, :, trials]
    print(f"K = {args.k}: {len(trials)} trials, {ts.groupby(['Shallow_Flow_Preset_Pump', 'Deep_Flow_Preset_Pump']).ngroups} conditions")

    pl = np.load(os.path.join(ROOT, "data", "pathlengths_paper_geom.npz"))
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}

    # Model A: fit on all K trials
    a, db0, res = calibrate(det, meta, beta, tau, g2, trials)
    print(f"Model A fit on {len(trials)} trials: a = {a:.4e}, D_B0 = {db0:.4e} "
          f"({res.nit} it / {res.nfev} evals, RMSE {np.sqrt(res.fun):.4f}, {time.time() - t0:.0f} s)", flush=True)
    sim = {"A": _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)}

    # held-out: fit on ranks 1..K-1, score rank K
    held = None
    if args.k > 1:
        tr = np.sort(ts.loc[ts["rank"] < args.k, "Trial_Index"].to_numpy(int))
        te = np.sort(ts.loc[ts["rank"] == args.k, "Trial_Index"].to_numpy(int))
        a_h, db0_h, _ = calibrate(det, meta, beta, tau, g2, tr)
        sim_te = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, te, a_h, db0_h)
        sim_in = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, te, a, db0)
        r2h = lambda s_: [float(np.median([_r2(s_[s, :, k], g2[s, :, te[k]]) for k in range(len(te))])) for s in range(3)]
        held = {"fit_trials": int(len(tr)), "test_trials": int(len(te)), "a": a_h, "db0": db0_h,
                "r2_median_heldout": r2h(sim_te), "r2_median_same_trials_fit_on_all": r2h(sim_in)}
        print(f"held-out: fit on {len(tr)} (ranks 1..{args.k - 1}) a = {a_h:.4e}, D_B0 = {db0_h:.4e} | "
              f"R2 on unseen rank-{args.k} trials {['%.3f' % x for x in held['r2_median_heldout']]} "
              f"(fit-on-all: {['%.3f' % x for x in held['r2_median_same_trials_fit_on_all']]})", flush=True)

    detp = wg.load_detp()
    detp_fixed = detp.astype(float).copy()
    detp_fixed[1:3] *= VOXEL_MM                   # pathlength rows: voxel -> mm
    sim["B"] = run_b(detp, meta, g2, trials, tau)
    sim["B fixed"] = run_b(detp_fixed, meta, g2, trials, tau)

    bfi_real = bfi_fit.fit_bfi_all(ref, beta, tau) * BFI_MM2S_TO_FILE
    bfi_y = meta["bfi"][:, trials]
    out = {"k": args.k, "n_trials": int(len(trials)), "a": a, "db0": db0, "held_out": held, "models": {}}
    r2_all, rx_all = {}, {}
    rows = {"Trial_Index": trials, "FR_set_s": meta["fr_set_s"][trials], "FR_set_d": meta["fr_set_d"][trials]}
    for name, s_ in sim.items():
        summ, r2, rx = score(s_, ref, beta, tau, bfi_real, bfi_y)
        out["models"][name] = summ
        r2_all[name], rx_all[name] = r2, rx
        for s, lab in enumerate(LAB):
            rows[f"r2_{name}_{lab}"] = r2[s]
            rows[f"bfiX_ratio_{name}_{lab}"] = rx[s]
        print(f"scored {name} ({time.time() - t0:.0f} s)", flush=True)

    wins = {}
    for x, y in (("A", "B"), ("A", "B fixed"), ("B fixed", "B")):
        r2w = [[int(np.sum(r2_all[x][s] > r2_all[y][s])), int(np.sum(r2_all[y][s] > r2_all[x][s]))] for s in range(3)]
        bw = [[int(np.sum(np.abs(np.log(rx_all[x][s])) < np.abs(np.log(rx_all[y][s])))),
               int(np.sum(np.abs(np.log(rx_all[y][s])) < np.abs(np.log(rx_all[x][s]))))] for s in range(3)]
        wins[f"{x} vs {y}"] = {"r2": r2w, "bfiX_log": bw}
    out["wins"] = wins
    fs = meta["fr_set_s"][trials]
    grp = np.select([fs <= 1, fs <= 5, fs <= 15], ["0-1", "2-5", "6-15"], ">=20")
    out["fixedB_wins_r2_by_shallow_group"] = {
        g: [int(np.sum((r2_all["B fixed"][s] > r2_all["A"][s]) & (grp == g))) for s in range(3)] + [int(np.sum(grp == g))]
        for g in ["0-1", "2-5", "6-15", ">=20"]}

    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "plots", f"k{args.k}_test_per_curve.csv"), index=False)
    with open(os.path.join(ROOT, "plots", f"k{args.k}_test.json"), "w") as f:
        json.dump(out, f, indent=2)

    fmt = lambda v, d=3: " / ".join(f"{x:.{d}f}" for x in v)
    for name, r in out["models"].items():
        print(f"\n[{name}]")
        print(f"  R2 median {fmt(r['r2_median'])}  mean {fmt(r['r2_mean'], 2)}  p10 {fmt(r['r2_p10'], 2)}")
        print(f"  RMSE pooled {fmt(r['rmse_pooled'], 4)}")
        print(f"  BFI-X ratio median {fmt(r['bfiX_ratio_median'], 2)}  mean {fmt(r['bfiX_ratio_mean'], 2)}  fold {fmt(r['bfiX_fold_error'], 2)}")
        print(f"  BFI-Y ratio median {fmt(r['bfiY_ratio_median'], 2)}  mean {fmt(r['bfiY_ratio_mean'], 2)}")
    for k_, v in wins.items():
        print(f"wins {k_}: R2 {v['r2']}  BFI-X(log) {v['bfiX_log']}")
    print("B fixed beats A (R2) by shallow group [1cm, 2cm, 3cm, n]:", out["fixedB_wins_r2_by_shallow_group"])
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
