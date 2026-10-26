"""
Per-trial metrics of Model A vs Model B on the closest-row test set (132 conditions), and for
each metric the win share and the best / worst condition of each model, per SDS.

Metrics (per trial, per SDS):
  RMSE of g2      sqrt(mean over 501 tau of (g2_sim - g2_real)^2)          lower is better
  BFI-X ratio     BFI-X(sim) / BFI-X(real g2), same fitter (bfi_fit.py)      closer to 1 is better
  BFI-Y ratio     BFI-X(sim) / BFI-Y (metadata.csv column, unknown algorithm) closer to 1 is better
  R^2             1 - SS_res/SS_tot of the g2 curve                          higher is better
Ratio wins are judged by |log(ratio)| (a factor 2 too high and 2 too low count the same).

  Model A  photons --photons-a (default paper geometry), (a, D_B0) from --cal-json
  Model B  Wichuda_r as her code runs it (../wichuda_g2): preset flow, 0.4e-6 * flow, beta per condition

Usage: .venv/bin/python scripts/ab_metric_extremes.py
Output: plots/ab_metric_per_trial.csv, plots/ab_metric_extremes.json
"""
import argparse
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

LAB = ["1cm", "2cm", "3cm"]
# metric -> (column stem, score used to rank: lower = better)
METRICS = {
    "RMSE of g2": ("rmse", lambda v: v),
    "BFI-X ratio": ("bfiX_ratio", lambda v: np.abs(np.log(v))),
    "BFI-Y ratio": ("bfiY_ratio", lambda v: np.abs(np.log(v))),
    "R2": ("r2", lambda v: -v),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons-a", default=os.path.join(ROOT, "data", "pathlengths_paper_geom.npz"))
    ap.add_argument("--cal-json", default=os.path.join(ROOT, "plots", "paper_geometry_compare.json"))
    ap.add_argument("--model-key", default="paper")
    ap.add_argument("--b-scale", type=float, default=0.4e-6)
    args = ap.parse_args()
    wg.FLOW_TO_ALPHA_DB_SCALE = args.b_scale

    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    trials = np.sort(pd.read_csv(os.path.join(ROOT, "dataset", "testset_closest.csv"))["Trial_Index"].to_numpy(int))
    ref = g2[:, :, trials]
    n = len(trials)

    cal = json.load(open(args.cal_json))["models"][args.model_key]
    pl = np.load(args.photons_a)
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
    sim = {"A": _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, cal["a"], cal["db0"])}

    detp = wg.load_detp()
    sim["B"] = np.zeros_like(ref)
    cache = {}
    for k, i in enumerate(trials):
        key = (meta["fr_set_s"][i], meta["fr_set_d"][i])
        if key not in cache:
            same = np.where((meta["fr_set_s"] == key[0]) & (meta["fr_set_d"] == key[1]))[0]
            b = [wg.beta_from_physical_trials(g2[s][:, same].T) for s in range(3)]
            cache[key] = wg.generate_g2(key[0], key[1], b, detp=detp, tau_seconds=tau)
        sim["B"][:, :, k] = cache[key]

    bfi_real = bfi_fit.fit_bfi_all(ref, beta, tau) * BFI_MM2S_TO_FILE
    bfi_y = meta["bfi"][:, trials]
    rows = {"Trial_Index": trials, "FR_set_s": meta["fr_set_s"][trials], "FR_set_d": meta["fr_set_d"][trials]}
    for m in ("A", "B"):
        bfi_sim = bfi_fit.fit_bfi_all(sim[m], beta, tau) * BFI_MM2S_TO_FILE
        for s, lab in enumerate(LAB):
            rows[f"rmse_{m}_{lab}"] = np.sqrt(np.mean((sim[m][s] - ref[s]) ** 2, axis=0))
            rows[f"r2_{m}_{lab}"] = np.array([_r2(sim[m][s, :, k], ref[s, :, k]) for k in range(n)])
            rows[f"bfiX_ratio_{m}_{lab}"] = bfi_sim[s] / bfi_real[s]
            rows[f"bfiY_ratio_{m}_{lab}"] = bfi_sim[s] / bfi_y[s]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "plots", "ab_metric_per_trial.csv"), index=False)

    cond = [f"({a:g},{b:g})" for a, b in zip(df.FR_set_s, df.FR_set_d)]
    out = {}
    for name, (stem, score) in METRICS.items():
        out[name] = {}
        print(f"\n== {name} ==")
        for lab in LAB:
            va, vb = df[f"{stem}_A_{lab}"].to_numpy(), df[f"{stem}_B_{lab}"].to_numpy()
            sa, sb = score(va), score(vb)
            res = {"A_wins": int(np.sum(sa < sb)), "B_wins": int(np.sum(sb < sa))}
            for m, v, sc in (("A", va, sa), ("B", vb, sb)):
                ib, iw = int(np.argmin(sc)), int(np.argmax(sc))
                res[m] = {"best": {"cond": cond[ib], "trial": int(trials[ib]), "value": float(v[ib])},
                          "worst": {"cond": cond[iw], "trial": int(trials[iw]), "value": float(v[iw])}}
            out[name][lab] = res
            print(f"  {lab}: wins A {res['A_wins']} - B {res['B_wins']} | "
                  f"A best {res['A']['best']['cond']} {res['A']['best']['value']:.3f} worst {res['A']['worst']['cond']} "
                  f"{res['A']['worst']['value']:.3f} | B best {res['B']['best']['cond']} {res['B']['best']['value']:.3f} "
                  f"worst {res['B']['worst']['cond']} {res['B']['worst']['value']:.3f}")
    with open(os.path.join(ROOT, "plots", "ab_metric_extremes.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved plots/ab_metric_per_trial.csv, plots/ab_metric_extremes.json")


if __name__ == "__main__":
    main()
