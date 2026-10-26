"""
Model B with Model A's flow -> alpha*D_B equation, everything else of B unchanged.

  A          our photons (paper geometry), FR_meas, alpha*D_B = a*FR + D_B0, beta per SDS
  B          Wichuda_r as her code runs it: her photons (pathlength in voxel units), preset flow,
             alpha*D_B = 0.4e-6 * FR (no D_B0), beta per condition
  B + eq A   B, but alpha*D_B = a*max(FR, 0) + D_B0 with A's (a, D_B0) as-is (no re-fit)
             -> isolates the effect of the flow -> alpha*D_B equation alone
  ... (FR_meas)  the same two B variants driven by the MEASURED outlet flow of each trial instead
             of the preset (negative measured flow clamped to 0; beta still per condition)

Scored on the closest-row test set (132 conditions) with the same metrics as the summary table:
R^2 median / mean, pooled RMSE of g2, BFI-X and BFI-Y ratio medians, per-curve R^2 wins.

Usage: .venv/bin/python scripts/b_with_a_equation.py
Output: plots/b_with_a_equation.json, plots/b_with_a_equation_per_trial.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bfi_fit
import data_io
from compare import _synth_curves, _r2, BFI_MM2S_TO_FILE
from compare_with_wichuda import wg

B_SCALE = 0.4e-6     # Wichuda_r commit 64985e7
LAB = ["1cm", "2cm", "3cm"]


def run_b(meta, g2, trials, tau, flow_rule, flow="set"):
    """Model B's g2 for every test trial, with the given flow -> alpha*D_B rule.
    flow="set": preset pump flow (as her code runs it); flow="meas": measured outlet flow,
    negative values clamped to 0. beta is always per condition (her convention)."""
    wg.flow_to_alpha_db = flow_rule
    detp = wg.load_detp()
    out = np.zeros((3, len(tau), len(trials)))
    beta_cache = {}
    for k, i in enumerate(trials):
        key = (meta["fr_set_s"][i], meta["fr_set_d"][i])
        if key not in beta_cache:
            same = np.where((meta["fr_set_s"] == key[0]) & (meta["fr_set_d"] == key[1]))[0]
            beta_cache[key] = [wg.beta_from_physical_trials(g2[s][:, same].T) for s in range(3)]
        if flow == "set":
            fs, fd = key
        else:
            fs, fd = max(meta["fr_meas_s"][i], 0.0), max(meta["fr_meas_d"][i], 0.0)
        out[:, :, k] = wg.generate_g2(fs, fd, beta_cache[key], detp=detp, tau_seconds=tau)
    return out


def main():
    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    trials = np.sort(pd.read_csv(os.path.join(ROOT, "dataset", "testset_closest.csv"))["Trial_Index"].to_numpy(int))
    ref = g2[:, :, trials]
    n = len(trials)

    cal = json.load(open(os.path.join(ROOT, "plots", "paper_geometry_compare.json")))["models"]["paper"]
    a, db0 = cal["a"], cal["db0"]
    pl = np.load(os.path.join(ROOT, "data", "pathlengths_paper_geom.npz"))
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}

    sims = {
        "A": _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0),
        "B": run_b(meta, g2, trials, tau, lambda f: float(f) * B_SCALE),
        "B + eq A": run_b(meta, g2, trials, tau, lambda f: a * max(float(f), 0.0) + db0),
        "B (FR_meas)": run_b(meta, g2, trials, tau, lambda f: float(f) * B_SCALE, flow="meas"),
        "B + eq A (FR_meas)": run_b(meta, g2, trials, tau, lambda f: a * max(float(f), 0.0) + db0,
                                    flow="meas"),
    }

    bfi_real = bfi_fit.fit_bfi_all(ref, beta, tau) * BFI_MM2S_TO_FILE
    bfi_y = meta["bfi"][:, trials]
    rows = {"Trial_Index": trials, "FR_set_s": meta["fr_set_s"][trials], "FR_set_d": meta["fr_set_d"][trials]}
    summary = {"a": a, "db0": db0, "n_trials": int(n), "models": {}}
    r2_all = {}
    for name, sim in sims.items():
        bfi_sim = bfi_fit.fit_bfi_all(sim, beta, tau) * BFI_MM2S_TO_FILE
        r2 = np.array([[_r2(sim[s, :, k], ref[s, :, k]) for k in range(n)] for s in range(3)])
        r2_all[name] = r2
        res = {"r2_median": [], "r2_mean": [], "rmse_pooled": [], "bfiX_ratio_median": [],
               "bfiX_ratio_mean": [], "bfiY_ratio_median": [], "bfiY_ratio_mean": []}
        for s, lab in enumerate(LAB):
            res["r2_median"].append(float(np.median(r2[s])))
            res["r2_mean"].append(float(np.mean(r2[s])))
            res["rmse_pooled"].append(float(np.sqrt(np.mean((sim[s] - ref[s]) ** 2))))
            res["bfiX_ratio_median"].append(float(np.median(bfi_sim[s] / bfi_real[s])))
            res["bfiX_ratio_mean"].append(float(np.mean(bfi_sim[s] / bfi_real[s])))
            res["bfiY_ratio_median"].append(float(np.median(bfi_sim[s] / bfi_y[s])))
            res["bfiY_ratio_mean"].append(float(np.mean(bfi_sim[s] / bfi_y[s])))
            rows[f"bfiY_ratio_{name}_{lab}"] = bfi_sim[s] / bfi_y[s]
            rows[f"r2_{name}_{lab}"] = r2[s]
            rows[f"bfiX_ratio_{name}_{lab}"] = bfi_sim[s] / bfi_real[s]
        summary["models"][name] = res

    wins = {}
    for x, y in (("A", "B + eq A"), ("B + eq A", "B"), ("B (FR_meas)", "B"),
                 ("B + eq A (FR_meas)", "B + eq A"), ("A", "B + eq A (FR_meas)")):
        wins[f"{x} vs {y}"] = [[int(np.sum(r2_all[x][s] > r2_all[y][s])), int(np.sum(r2_all[y][s] > r2_all[x][s]))]
                              for s in range(3)]
    summary["r2_wins"] = wins

    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "plots", "b_with_a_equation_per_trial.csv"), index=False)
    with open(os.path.join(ROOT, "plots", "b_with_a_equation.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fmt = lambda v, d=3: " / ".join(f"{x:.{d}f}" for x in v)
    print(f"A's equation: alpha*D_B = {a:.4e} * FR + {db0:.4e}")
    for name, r in summary["models"].items():
        print(f"\n[{name}]")
        print(f"  R2 median   {fmt(r['r2_median'])}")
        print(f"  R2 mean     {fmt(r['r2_mean'], 2)}")
        print(f"  RMSE pooled {fmt(r['rmse_pooled'])}")
        print(f"  BFI-X ratio median {fmt(r['bfiX_ratio_median'], 2)}  mean {fmt(r['bfiX_ratio_mean'], 2)}")
        print(f"  BFI-Y ratio median {fmt(r['bfiY_ratio_median'], 2)}  mean {fmt(r['bfiY_ratio_mean'], 2)}")
    for k, v in wins.items():
        print(f"R2 wins {k}: " + "  ".join(f"{LAB[s]} {w[0]}-{w[1]}" for s, w in enumerate(v)))
    print("saved plots/b_with_a_equation.json, plots/b_with_a_equation_per_trial.csv")


if __name__ == "__main__":
    main()
