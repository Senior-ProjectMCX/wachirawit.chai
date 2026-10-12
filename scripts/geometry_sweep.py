"""
Geometry optimisation runs (plan 2026-10-12, notes in Obsidian "Geometry Optimization Research").
One configuration per call; every number comes from the paper or a fibre datasheet:

  --deep-mm 35        deep layer 35 mm (Nakabayashi Sec. 2.1) instead of 44.5 mm
  --cone              source = cone of half-angle asin(0.39/1.37) ~ 16.5 deg (FT400EMT, NA 0.39)
  --na                keep only photons inside the detector fibre's acceptance,
                      sin(theta_air) <= 0.12 (S630-HP, NA 0.12)
  --ring N            N detectors on a ring per SDS (pooled) to recover photon counts after --na
  --shallow-mm X      shallow layer thickness (paper's own sensitivity range 4.5-6.5 mm, Sec. 2.4.3)

Scoring = Compare Method (Simple) on the 108-trial test set (dataset/testset_closest.csv):
(a, D_B0) calibrated on those trials, RMSE / R2 per trial / BFI-A, plus the c2/c1 gate
(paper target 0.140/0.385/0.459 and our fitter on the physical g2 of the same trials), and a
per-trial head-to-head against G0 (no averaging of RMSE).

MCX output is cached in data/geom_<name>.npz, results in plots/geometry_sweep/<name>.json.

Usage:
  .venv/bin/python scripts/geometry_sweep.py --name G0 --baseline
  .venv/bin/python scripts/geometry_sweep.py --name G1 --deep-mm 35
  .venv/bin/python scripts/geometry_sweep.py --name G3 --na --ring 8 --nphoton 4e8
  .venv/bin/python scripts/geometry_sweep.py --summary
"""
import argparse
import glob
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
from config import parameters as cfg
import bfi_fit
import data_io
from compare import _synth_curves, _r2, CAL_TAU_STRIDE, BFI_MM2S_TO_FILE

OUT_DIR = os.path.join(ROOT, "plots", "geometry_sweep")
MAX_PHOTONS = 5000


def run_mcx(args):
    path = os.path.join(ROOT, "data", f"geom_{args.name}.npz")
    if args.baseline:
        return np.load(os.path.join(ROOT, "data", "pathlengths.npz")), "data/pathlengths.npz (existing)"
    if os.path.exists(path):
        return np.load(path), path
    import mcx_engine
    saved = (cfg.SLAB_Z_MM, cfg.MC_SHALLOW_DEPTH_MM)
    cfg.MC_SHALLOW_DEPTH_MM = args.shallow_mm
    cfg.SLAB_Z_MM = args.shallow_mm + args.deep_mm
    t0 = time.time()
    paths = mcx_engine.run_simulation(
        nphoton=args.nphoton,
        srctype="cone" if args.cone else "pencil",
        src_half_angle_rad=cfg.SOURCE_HALF_ANGLE_RAD if args.cone else 0.0,
        save_exit_dir=args.na, n_ring=args.ring)
    cfg.SLAB_Z_MM, cfg.MC_SHALLOW_DEPTH_MM = saved
    np.savez_compressed(path, **paths)
    print(f"[{args.name}] MCX {args.nphoton:.1e} photons in {time.time() - t0:.0f}s -> {path}", flush=True)
    return np.load(path), path


def detectors(pl, use_na):
    det, counts = {}, {}
    rng = np.random.default_rng(42)
    for s in range(3):
        k = s + 1
        arrs = [pl[f"det_{k}_{x}"] for x in ("Ls", "Ld", "Ys", "Yd")]
        n_all = len(arrs[0])
        if use_na:
            keep = pl[f"det_{k}_cos"] >= np.cos(cfg.DETECTOR_HALF_ANGLE_AIR_RAD)
            arrs = [a[keep] for a in arrs]
        n = len(arrs[0])
        if n > MAX_PHOTONS:
            idx = rng.choice(n, MAX_PHOTONS, replace=False)
            arrs = [a[idx] for a in arrs]
        det[s] = tuple(arrs)
        counts[data_io.SDS_LABELS[s]] = {"detected": int(n_all), "used": int(n), "after_na": int(n) if use_na else None}
    return det, counts


def evaluate(det, meta, g2, beta, tau, trials):
    fr_s, fr_d = meta["fr_meas_s"], meta["fr_meas_d"]
    tc = tau[::CAL_TAU_STRIDE]
    ref_c = g2[:, ::CAL_TAU_STRIDE, :][:, :, trials]

    def loss(p):
        return np.mean((_synth_curves(det, fr_s, fr_d, beta, tc, trials, 10 ** p[0], 10 ** p[1]) - ref_c) ** 2)

    r = minimize(loss, [np.log10(cfg.DEFAULT_A_MM2_S_PER_FR), np.log10(cfg.DEFAULT_DB0_MM2_S)],
                 method="Nelder-Mead", bounds=[(-9.0, -3.0), (-9.0, -5.0)])
    a, db0 = 10 ** r.x[0], 10 ** r.x[1]
    sim = _synth_curves(det, fr_s, fr_d, beta, tau, trials, a, db0)
    ref = g2[:, :, trials]
    bfi_sim = bfi_fit.fit_bfi_all(sim, beta, tau) * BFI_MM2S_TO_FILE
    return a, db0, sim, ref, bfi_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--baseline", action="store_true", help="G0: reuse data/pathlengths.npz")
    ap.add_argument("--deep-mm", type=float, default=cfg.SLAB_Z_MM - cfg.MC_SHALLOW_DEPTH_MM)
    ap.add_argument("--shallow-mm", type=float, default=cfg.MC_SHALLOW_DEPTH_MM)
    ap.add_argument("--cone", action="store_true")
    ap.add_argument("--na", action="store_true")
    ap.add_argument("--ring", type=int, default=1)
    ap.add_argument("--nphoton", type=float, default=1e8)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.summary:
        return summary()

    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    trials = np.sort(pd.read_csv(os.path.join(ROOT, "dataset", "testset_closest.csv"))["Trial_Index"].to_numpy(int))

    pl, src = run_mcx(args)
    det, counts = detectors(pl, args.na)
    print(f"[{args.name}] photons per SDS: {counts}", flush=True)

    a, db0, sim, ref, bfi_sim = evaluate(det, meta, g2, beta, tau, trials)
    bfi_phys = bfi_fit.fit_bfi_all(ref, beta, tau) * BFI_MM2S_TO_FILE
    gate_sim = [float(r[4]) for r in bfi_fit.regress_c2c1(bfi_sim, meta["fr_meas_s"][trials], meta["fr_meas_d"][trials])]
    gate_phys = [float(r[4]) for r in bfi_fit.regress_c2c1(bfi_phys, meta["fr_meas_s"][trials], meta["fr_meas_d"][trials])]

    per_trial_rmse = np.sqrt(np.mean((sim - ref) ** 2, axis=1))          # (3, n)
    rows = []
    for s in range(3):
        r2t = [_r2(sim[s][:, j], ref[s][:, j]) for j in range(len(trials))]
        rows.append({"sds": data_io.SDS_LABELS[s],
                     "rmse": float(np.sqrt(np.mean((sim[s] - ref[s]) ** 2))),
                     "r2_per_trial_median": float(np.median(r2t)),
                     "bfi_A_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_phys[s])[0, 1]),
                     "bfi_A_median_ratio": float(np.median(bfi_sim[s] / bfi_phys[s])),
                     "c2c1_sim": gate_sim[s], "c2c1_phys_same_fitter": gate_phys[s],
                     "c2c1_paper": float(cfg.TARGET_C2_C1[s])})
    out = {"name": args.name, "source": src,
           "config": {"deep_mm": args.deep_mm, "shallow_mm": args.shallow_mm, "cone": args.cone,
                      "na": args.na, "ring": args.ring, "nphoton": args.nphoton, "baseline": args.baseline},
           "photons": counts, "a": a, "db0": db0, "rows": rows,
           "per_trial_rmse": per_trial_rmse.tolist(), "trials": trials.tolist()}
    with open(os.path.join(OUT_DIR, f"{args.name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    for r in rows:
        print(f"[{args.name}] {r['sds']}: RMSE={r['rmse']:.4f} R2/trial={r['r2_per_trial_median']:.3f} "
              f"BFI-A R={r['bfi_A_pearson_r']:.3f} ratio={r['bfi_A_median_ratio']:.2f} | "
              f"c2/c1 sim={r['c2c1_sim']:.3f} phys={r['c2c1_phys_same_fitter']:.3f} paper={r['c2c1_paper']:.3f}",
              flush=True)


def summary():
    res = {}
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.json"))):
        d = json.load(open(f))
        res[d["name"]] = d
    if "G0" not in res:
        print("run G0 first")
        return
    base = np.array(res["G0"]["per_trial_rmse"])
    print(f"{'step':<6}{'R2/trial 1/2/3cm':>22}{'RMSE 1/2/3cm':>24}{'c2/c1 sim':>22}"
          f"{'wins vs G0 (trial x SDS)':>28}  photons used")
    for name, d in res.items():
        r = d["rows"]
        pt = np.array(d["per_trial_rmse"])
        wins = int((pt < base).sum()) if name != "G0" else 0
        losses = int((pt > base).sum()) if name != "G0" else 0
        used = "/".join(str(d["photons"][k]["used"]) for k in ("1cm", "2cm", "3cm"))
        r2s = "/".join(f"{x['r2_per_trial_median']:.3f}" for x in r)
        rms = "/".join(f"{x['rmse']:.4f}" for x in r)
        gate = "/".join(f"{x['c2c1_sim']:.3f}" for x in r)
        wl = f"{wins} - {losses}" if name != "G0" else "-"
        print(f"{name:<6}{r2s:>22}{rms:>24}{gate:>22}{wl:>28}  {used}")
    r0 = res["G0"]["rows"]
    print("c2/c1 targets: paper " + "/".join(f"{x['c2c1_paper']:.3f}" for x in r0)
          + "   physical same-fitter " + "/".join(f"{x['c2c1_phys_same_fitter']:.3f}" for x in r0))


if __name__ == "__main__":
    main()
