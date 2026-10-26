"""
g2 of selected conditions, SDS 1/2/3 cm side by side: real vs Model A vs Model B.
Default picks (from plots/g2_overlay_testset_paper_geom/r2_per_curve.csv, R^2 per curve):
  A wins on all 3 SDS, spread over flow:  (0, 5) trial 252, (10, 10) trial 786, (70, 70) trial 1536
  B wins on all 3 SDS (the only three):   (5, 20) trial 668, (5, 40) trial 683, (5, 60) trial 704

  Model A  photons --photons-a (default: paper geometry), (a, D_B0) read from --cal-json
           (scripts/paper_geometry_compare.py output, so no re-calibration), FR_meas, beta per SDS
  Model B  Wichuda_r as her code runs it (../wichuda_g2): preset flow, 0.4e-6 * flow, beta per condition
Titles give R^2 (1 - SS_res/SS_tot) and BFI-X (our fitter, 1e-9 cm^2/s) of real / A / B.

Usage: .venv/bin/python scripts/plot_g2_win_cases.py
Output: plots/g2_win_cases_paper_geom.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bfi_fit
import data_io
from compare import _synth_curves, _r2, BFI_MM2S_TO_FILE
from compare_with_wichuda import wg

A_WINS = [252, 786, 1536]
B_WINS = [668, 683, 704]
SDS = ["1 cm", "2 cm", "3 cm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photons-a", default=os.path.join(ROOT, "data", "pathlengths_paper_geom.npz"))
    ap.add_argument("--cal-json", default=os.path.join(ROOT, "plots", "paper_geometry_compare.json"))
    ap.add_argument("--model-key", default="paper", help="model entry in --cal-json")
    ap.add_argument("--b-scale", type=float, default=0.4e-6)
    ap.add_argument("--out", default=os.path.join(ROOT, "plots", "g2_win_cases_paper_geom.png"))
    args = ap.parse_args()
    wg.FLOW_TO_ALPHA_DB_SCALE = args.b_scale

    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    tau_us = tau * 1e6
    trials = np.array(A_WINS + B_WINS)
    ref = g2[:, :, trials]

    cal = json.load(open(args.cal_json))["models"][args.model_key]
    a, db0 = cal["a"], cal["db0"]
    pl = np.load(args.photons_a)
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
    sim_a = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)

    detp = wg.load_detp()
    sim_b = np.zeros_like(ref)
    for k, i in enumerate(trials):
        key = (meta["fr_set_s"][i], meta["fr_set_d"][i])
        same = np.where((meta["fr_set_s"] == key[0]) & (meta["fr_set_d"] == key[1]))[0]
        b = [wg.beta_from_physical_trials(g2[s][:, same].T) for s in range(3)]
        sim_b[:, :, k] = wg.generate_g2(key[0], key[1], b, detp=detp, tau_seconds=tau)

    bfi = {name: bfi_fit.fit_bfi_all(x, beta, tau) * BFI_MM2S_TO_FILE
           for name, x in (("real", ref), ("A", sim_a), ("B", sim_b))}

    m = tau_us > 0
    fig, axes = plt.subplots(len(trials), 3, figsize=(13, 2.55 * len(trials)), sharex=True)
    for k, i in enumerate(trials):
        for s in range(3):
            ax = axes[k, s]
            ax.semilogx(tau_us[m], ref[s, m, k], color="black", lw=2.2, label="real")
            ax.semilogx(tau_us[m], sim_a[s, m, k], color="tab:blue", lw=1.6, label="Model A")
            ax.semilogx(tau_us[m], sim_b[s, m, k], color="tab:red", lw=1.6, ls="--", label="Model B")
            r2a, r2b = _r2(sim_a[s, :, k], ref[s, :, k]), _r2(sim_b[s, :, k], ref[s, :, k])
            ax.set_title(f"SDS {SDS[s]}   R² A {r2a:.2f} | B {r2b:.2f}\n"
                         f"BFI-X real {bfi['real'][s, k]:.1f} | A {bfi['A'][s, k]:.1f} | B {bfi['B'][s, k]:.1f}",
                         fontsize=8)
            ax.set_xlim(1, 500)
            ax.grid(alpha=0.3, which="both", lw=0.4)
            ax.tick_params(labelsize=7)
        axes[k, 0].set_ylabel(f"trial {i}\nset ({meta['fr_set_s'][i]:g}, {meta['fr_set_d'][i]:g})\n"
                              f"meas ({meta['fr_meas_s'][i]:.1f}, {meta['fr_meas_d'][i]:.1f})\ng2", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("τ (µs)", fontsize=8)
    axes[0, 2].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"g2: real vs Model A vs Model B  "
                 f"(A: a={a:.3e}, D_B0={db0:.3e}; B: {args.b_scale:.1e} x preset flow)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")
    for k, i in enumerate(trials):
        print(f"trial {i} ({meta['fr_set_s'][i]:g},{meta['fr_set_d'][i]:g}): "
              + "  ".join(f"{SDS[s]} R2 A {_r2(sim_a[s, :, k], ref[s, :, k]):.3f} B {_r2(sim_b[s, :, k], ref[s, :, k]):.3f} "
                          f"BFI-X {bfi['real'][s, k]:.1f}/{bfi['A'][s, k]:.1f}/{bfi['B'][s, k]:.1f}" for s in range(3)))


if __name__ == "__main__":
    main()
