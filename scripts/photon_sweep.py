"""
How many MCX photons does Model A need? Re-run MCX (paper geometry) at several launched-photon
counts, time each run, and score each photon set on the K-closest test set with the SAME
(a, D_B0) (taken from plots/k{K}_test.json, fitted on the 2e8 set), so only photon noise differs.

Per photon count: MCX wall time, detected photons per SDS, photons actually used (5,000 per SDS
cap, as in every other script), R^2 median / mean, pooled RMSE of g2, BFI-X ratio median, and the
largest |g2 - g2(2e8)| over all trials (how far the curve moves from the reference set).

Usage: .venv/bin/python scripts/photon_sweep.py [--k 5] [--counts 1e7 2e7 5e7 1e8]
Output: data/pathlengths_paper_geom_n<count>.npz, plots/photon_sweep_k{K}.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
import bfi_fit
import data_io
import mcx_engine
from compare import _synth_curves, _r2, BFI_MM2S_TO_FILE

REF_FILE = os.path.join(ROOT, "data", "pathlengths_paper_geom.npz")     # 2e8, the current Model A set
REF_SECONDS = 219.3                                                     # timed MCX run of that set (2026-10-15)


def evaluate(pl, a, db0, meta, g2, beta, tau, trials, bfi_real):
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
    used = [int(len(det[s][0])) for s in range(3)]
    t0 = time.time()
    sim = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)
    sim_sec = time.time() - t0
    ref = g2[:, :, trials]
    r2 = np.array([[_r2(sim[s, :, k], ref[s, :, k]) for k in range(len(trials))] for s in range(3)])
    bfi = bfi_fit.fit_bfi_all(sim, beta, tau) * BFI_MM2S_TO_FILE
    return sim, {
        "photons_used": used,
        "g2_sim_seconds": round(sim_sec, 1),
        "r2_median": [float(np.median(r2[s])) for s in range(3)],
        "r2_mean": [float(np.mean(r2[s])) for s in range(3)],
        "rmse_pooled": [float(np.sqrt(np.mean((sim[s] - ref[s]) ** 2))) for s in range(3)],
        "bfiX_ratio_median": [float(np.median(bfi[s] / bfi_real[s])) for s in range(3)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--counts", nargs="+", type=float, default=[1e7, 2e7, 5e7, 1e8])
    args = ap.parse_args()

    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    trials = np.sort(pd.read_csv(os.path.join(ROOT, "dataset", f"testset_closest_k{args.k}.csv"))["Trial_Index"].to_numpy(int))
    fit = json.load(open(os.path.join(ROOT, "plots", f"k{args.k}_test.json")))
    a, db0 = fit["a"], fit["db0"]
    bfi_real = bfi_fit.fit_bfi_all(g2[:, :, trials], beta, tau) * BFI_MM2S_TO_FILE
    print(f"K = {args.k}: {len(trials)} trials, a = {a:.4e}, D_B0 = {db0:.4e}", flush=True)

    pl_ref = np.load(REF_FILE)
    sim_ref, res_ref = evaluate(pl_ref, a, db0, meta, g2, beta, tau, trials, bfi_real)
    res_ref.update({"photons": 2e8, "mcx_seconds": REF_SECONDS,
                    "detected": [int(len(pl_ref[f"det_{s + 1}_Ls"])) for s in range(3)],
                    "max_abs_g2_diff_vs_2e8": [0.0, 0.0, 0.0]})
    results = []
    for n in args.counts:
        path = os.path.join(ROOT, "data", f"pathlengths_paper_geom_n{n:.0e}.npz".replace("+", ""))
        t0 = time.time()
        pl = mcx_engine.run_simulation(nphoton=n, geometry="paper")
        mcx_sec = time.time() - t0
        np.savez_compressed(path, **pl)
        sim, res = evaluate(pl, a, db0, meta, g2, beta, tau, trials, bfi_real)
        res.update({"photons": n, "mcx_seconds": round(mcx_sec, 1),
                    "detected": [int(len(pl[f"det_{s + 1}_Ls"])) for s in range(3)],
                    "max_abs_g2_diff_vs_2e8": [float(np.abs(sim[s] - sim_ref[s]).max()) for s in range(3)]})
        results.append(res)
        print(f"{n:.0e}: MCX {mcx_sec:.0f} s, detected {res['detected']}, R2 median "
              f"{['%.3f' % x for x in res['r2_median']]}, max|dg2| {['%.4f' % x for x in res['max_abs_g2_diff_vs_2e8']]}",
              flush=True)
    results.append(res_ref)
    with open(os.path.join(ROOT, "plots", f"photon_sweep_k{args.k}.json"), "w") as f:
        json.dump({"k": args.k, "a": a, "db0": db0, "runs": results}, f, indent=2)
    print("saved", os.path.join(ROOT, "plots", f"photon_sweep_k{args.k}.json"))


if __name__ == "__main__":
    main()
