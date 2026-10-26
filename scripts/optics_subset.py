"""
Optics hypothesis test on the stratified subset (follow-up to subset_experiment.py):
the shallow-low/deep-high corner fails at SDS 1cm even with a grid-covering
calibration, and the geometry gate (c2/c1) fails. Hypothesis: the MC medium's
mu_s'/mu_a should be lower than the nominal fluid value (glass beads dilute it).

For each optics fraction f: run MCX with mu_a, mu_s' = f * nominal (cached to
data/pathlengths_optics_<f>.npz; f=1.0 reuses data/pathlengths.npz), synthesise g2
with the same reduced mu_a, keep the BFI fitter at the paper's NOMINAL optics
(as the paper's own analysis would), then report per seed:
  - E1 held-out metrics (RMSE, R2/trial, BFI-X R) on the 4x4 stratified subset
  - the shallow-low/deep-high corner (R2, RMSE) with the grid-covering calibration
  - c2/c1 of synthetic BFI vs the physical same-fitter c2/c1 (geometry gate)

Usage: .venv/bin/python scripts/optics_subset.py [--fracs 1.0 0.8 0.6 0.4] [--nphoton 1e8]
"""
import argparse
import functools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from config import parameters as cfg
import bfi_fit
import compare
import data_io
import mcx_engine
import subset_experiment as se
from dcs_physics import calculate_g2 as _g2

NOM_MUS, NOM_MUA = cfg.MU_S_MM, cfg.MU_A_MM
SDS = data_io.SDS_LABELS


def get_pathlengths(frac, nphoton, outdir):
    if frac == 1.0:
        return np.load(os.path.join(ROOT, "data", "pathlengths.npz"))
    path = os.path.join(outdir, f"pathlengths_optics_{frac}.npz")
    if not os.path.exists(path):
        cfg.MU_S_MM, cfg.MU_A_MM = NOM_MUS * frac, NOM_MUA * frac
        t0 = time.time()
        paths = mcx_engine.run_simulation(nphoton=nphoton)
        cfg.MU_S_MM, cfg.MU_A_MM = NOM_MUS, NOM_MUA
        np.savez_compressed(path, **paths)
        print(f"[optics x{frac}] MCX done in {time.time() - t0:.0f}s -> {path}", flush=True)
    return np.load(path)


def c2c1(bfi, fr_s, fr_d):
    return [float(r[4]) for r in bfi_fit.regress_c2c1(bfi, fr_s, fr_d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--nphoton", type=float, default=1e8)
    ap.add_argument("--per-combo", type=int, default=6)
    ap.add_argument("--per-cell", type=int, default=3)
    args = ap.parse_args()

    meta = data_io.load_physical_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    tau = data_io.TAU_ARRAY
    fr_s, fr_d = meta["fr_meas_s"], meta["fr_meas_d"]
    outdir = os.path.join(ROOT, "data")

    # physical same-fitter c2/c1 on the union of selected trials (reference for the gate)
    designs = {}
    for seed in args.seeds:
        sel = se.pick_cells(se.build_design(meta, args.per_combo, seed), args.per_cell)
        designs[seed] = (sel, *se.split_e1(sel, seed))
    union = np.unique(np.concatenate([se.trials_of([c for cs in d[0].values() for c in cs])
                                      for d in designs.values()]))
    phys_bfi = bfi_fit.fit_bfi_all(phys_g2[:, :, union], beta, tau)
    phys_c2c1 = c2c1(phys_bfi, fr_s[union], fr_d[union])
    print(f"physical same-fitter c2/c1 on {len(union)} subset trials: "
          + " / ".join(f"{v:.3f}" for v in phys_c2c1), flush=True)

    results = {"physical_c2c1": phys_c2c1, "fracs": {}}
    for frac in args.fracs:
        pl = get_pathlengths(frac, args.nphoton, outdir)
        rng = np.random.default_rng(42)
        det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
        counts = [int(len(pl[f"det_{s + 1}_Ls"])) for s in range(3)]
        mua = NOM_MUA * frac
        compare.calculate_g2 = functools.partial(_g2, mua_s=mua, mua_d=mua)   # MC-consistent absorption
        ev = se.Evaluator(det, meta, phys_g2, beta, tau)
        per_seed = []
        for seed in args.seeds:
            sel, train, test = designs[seed]
            a, db0 = se.calibrate(det, fr_s, fr_d, beta, tau, phys_g2, train)
            overall = ev.run(se.trials_of(test), a, db0)
            corner = [c for cs in sel.values() for c in cs if c["sb"] == 0 and c["db"] == se.N_BINS - 1]
            corner_rows = ev.run(se.trials_of(corner), a, db0)
            sel_trials = se.trials_of([c for cs in sel.values() for c in cs])
            sim = compare._synth_curves(det, fr_s, fr_d, beta, tau, sel_trials, a, db0)
            gate = c2c1(bfi_fit.fit_bfi_all(sim, beta, tau), fr_s[sel_trials], fr_d[sel_trials])
            per_seed.append({"seed": seed, "a": a, "db0": db0, "overall": overall,
                             "corner": corner_rows, "c2c1": gate})
            print(f"[x{frac} seed {seed}] R2/trial " + "/".join(f"{r['r2_per_trial_median']:.3f}" for r in overall)
                  + " | RMSE " + "/".join(f"{r['rmse']:.4f}" for r in overall)
                  + " | corner R2 " + "/".join(f"{r['r2_per_trial_median']:.2f}" for r in corner_rows)
                  + " | c2/c1 " + "/".join(f"{v:.3f}" for v in gate), flush=True)
        results["fracs"][str(frac)] = {"photons_detected": counts, "per_seed": per_seed}
    compare.calculate_g2 = _g2

    # summary
    print("\n=========== SUMMARY (mean over seeds) ===========")
    print(f"physical c2/c1: " + " / ".join(f"{v:.3f}" for v in phys_c2c1) + "   (gate tolerance +-0.06)")
    print(f"{'optics':<8}{'R2/trial 1/2/3cm':>22}{'RMSE 1/2/3cm':>22}{'corner R2 (1cm)':>17}{'c2/c1 1/2/3cm':>24}")
    for frac, v in results["fracs"].items():
        ps = v["per_seed"]
        r2 = np.mean([[r["r2_per_trial_median"] for r in p["overall"]] for p in ps], axis=0)
        rm = np.mean([[r["rmse"] for r in p["overall"]] for p in ps], axis=0)
        cr = np.mean([p["corner"][0]["r2_per_trial_median"] for p in ps])
        g = np.mean([p["c2c1"] for p in ps], axis=0)
        v["summary"] = {"r2": r2.tolist(), "rmse": rm.tolist(), "corner_r2_1cm": float(cr), "c2c1": g.tolist()}
        print(f"x{frac:<7}{'/'.join(f'{x:.3f}' for x in r2):>22}{'/'.join(f'{x:.4f}' for x in rm):>22}"
              f"{cr:>17.2f}{'/'.join(f'{x:.3f}' for x in g):>24}")
    with open(os.path.join(ROOT, "plots", "optics_subset.json"), "w") as f:
        json.dump({"args": vars(args), **results}, f, indent=2)


if __name__ == "__main__":
    main()
