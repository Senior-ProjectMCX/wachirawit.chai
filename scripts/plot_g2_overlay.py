"""
Overlay plots on the closest-row test set (dataset/testset_closest.csv, 132 conditions,
(0,0) excluded): physical g2 vs Model A (ours) vs Model B (Wichuda_r) at SDS 1/2/3 cm.

  Model A  our MCX photons (data/pathlengths_shallow5.5.npz, main geometry), FR_meas,
           (a, D_B0) calibrated on the test set, beta per SDS
  Model B  Wichuda's photons and rules as her code runs them (../wichuda_g2): preset flow,
           alpha*D_B = SCALE * flow (no D_B0), beta per condition. SCALE defaults to 0.4e-6,
           her current value (Wichuda_r commit 64985e7); --b-scale 1e-6 gives the old one.

Outputs (plots/g2_overlay_testset/):
  overview.png         9 representative conditions x 3 SDS
  all_conditions.pdf   one page per condition (132 pages)
  cond_<trial>.png     one image per condition
  r2_per_curve.csv     R^2 (1 - SS_res/SS_tot) and BFI-X (real / A / B, 1e-9 cm^2/s, same
                       semi-infinite fitter) for every condition x SDS

Usage: .venv/bin/python scripts/plot_g2_overlay.py [--b-scale 0.4e-6] [--no-png]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bfi_fit
import data_io
from compare import _synth_curves, _r2, BFI_MM2S_TO_FILE
from compare_with_wichuda import calibrate, wg

OUT_DIR = os.path.join(ROOT, "plots", "g2_overlay_testset")
PHOTONS_A = os.path.join(ROOT, "data", "pathlengths_shallow5.5.npz")
SDS = ["1 cm", "2 cm", "3 cm"]
C_REAL, C_A, C_B = "black", "tab:blue", "tab:red"
LABEL_A = "A (wachirawit)"


def draw_condition(axes, k, tau_us, ref, sim_a, sim_b, meta, trial, r2a, r2b, show_legend):
    m = tau_us > 0
    for s, ax in enumerate(axes):
        ax.semilogx(tau_us[m], ref[s, m, k], color=C_REAL, lw=2.2, label="real")
        ax.semilogx(tau_us[m], sim_a[s, m, k], color=C_A, lw=1.5, label=LABEL_A)
        ax.semilogx(tau_us[m], sim_b[s, m, k], color=C_B, lw=1.5, ls="--", label="B (wichuda)")
        ax.set_title(f"SDS {SDS[s]}  R² A {r2a[s, k]:.3f} | B {r2b[s, k]:.3f}", fontsize=8)
        ax.set_xlim(1, 500)
        ax.grid(alpha=0.3, which="both", lw=0.4)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"trial {trial}\nset ({meta['fr_set_s'][trial]:g}, {meta['fr_set_d'][trial]:g})\n"
                       f"meas ({meta['fr_meas_s'][trial]:.1f}, {meta['fr_meas_d'][trial]:.1f})\ng2",
                       fontsize=7)
    if show_legend:
        axes[-1].legend(fontsize=7, loc="upper right")


def main():
    global OUT_DIR, LABEL_A
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-scale", type=float, default=0.4e-6, help="Model B flow -> alpha*D_B scale")
    ap.add_argument("--no-png", action="store_true", help="skip the per-condition PNGs")
    ap.add_argument("--photons-a", default=PHOTONS_A, help="Model A MCX photon file")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--label-a", default="A (wachirawit)")
    args = ap.parse_args()
    wg.FLOW_TO_ALPHA_DB_SCALE = args.b_scale
    OUT_DIR, LABEL_A = args.out_dir, args.label_a
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    tau_us = tau * 1e6
    trials = np.sort(pd.read_csv(os.path.join(ROOT, "dataset", "testset_closest.csv"))["Trial_Index"].to_numpy(int))
    ref = g2[:, :, trials]
    n = len(trials)

    # Model A
    pl = np.load(args.photons_a)
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
    a, db0 = calibrate(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, g2, trials)
    sim_a = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)
    print(f"Model A: a = {a:.3e}, D_B0 = {db0:.3e}")

    # Model B (as her code runs it)
    detp = wg.load_detp()
    sim_b = np.zeros_like(ref)
    cache = {}
    for k, i in enumerate(trials):
        key = (meta["fr_set_s"][i], meta["fr_set_d"][i])
        if key not in cache:
            same = np.where((meta["fr_set_s"] == key[0]) & (meta["fr_set_d"] == key[1]))[0]
            b = [wg.beta_from_physical_trials(g2[s][:, same].T) for s in range(3)]
            cache[key] = wg.generate_g2(key[0], key[1], b, detp=detp, tau_seconds=tau)
        sim_b[:, :, k] = cache[key]
    print(f"Model B: scale = {args.b_scale:.2e} (no D_B0)")

    r2a = np.array([[_r2(sim_a[s, :, k], ref[s, :, k]) for k in range(n)] for s in range(3)])
    r2b = np.array([[_r2(sim_b[s, :, k], ref[s, :, k]) for k in range(n)] for s in range(3)])
    # BFI-X: every curve fitted with the same semi-infinite fitter (Eq. 3), units 1e-9 cm^2/s
    bfi_real = bfi_fit.fit_bfi_all(ref, beta, tau) * BFI_MM2S_TO_FILE
    bfi_a = bfi_fit.fit_bfi_all(sim_a, beta, tau) * BFI_MM2S_TO_FILE
    bfi_b = bfi_fit.fit_bfi_all(sim_b, beta, tau) * BFI_MM2S_TO_FILE
    rows = []
    for k, i in enumerate(trials):
        row = {"Trial_Index": int(i), "FR_set_s": meta["fr_set_s"][i], "FR_set_d": meta["fr_set_d"][i],
               "FR_meas_s": meta["fr_meas_s"][i], "FR_meas_d": meta["fr_meas_d"][i]}
        for s, lab in enumerate(["1cm", "2cm", "3cm"]):
            row[f"R2_A_{lab}"], row[f"R2_B_{lab}"] = r2a[s, k], r2b[s, k]
            row[f"BFI_real_{lab}"], row[f"BFI_A_{lab}"], row[f"BFI_B_{lab}"] = bfi_real[s, k], bfi_a[s, k], bfi_b[s, k]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "r2_per_curve.csv"), index=False)
    for s in range(3):
        print(f"  SDS {SDS[s]}: R2 median A {np.median(r2a[s]):.3f}  B {np.median(r2b[s]):.3f}  "
              f"A better in {np.sum(r2a[s] > r2b[s])}/{n}")

    # overview: 9 conditions spread over the (shallow, deep) preset grid
    picks = [(1, 1), (1, 20), (1, 60), (20, 1), (20, 20), (20, 60), (60, 1), (60, 20), (60, 60)]
    fs, fd = meta["fr_set_s"][trials], meta["fr_set_d"][trials]
    idx = [int(np.argmin((fs - ps) ** 2 + (fd - pd_) ** 2)) for ps, pd_ in picks]
    fig, axes = plt.subplots(len(idx), 3, figsize=(12, 2.3 * len(idx)), sharex=True)
    for r, k in enumerate(idx):
        draw_condition(axes[r], k, tau_us, ref, sim_a, sim_b, meta, trials[k], r2a, r2b, r == 0)
    for ax in axes[-1]:
        ax.set_xlabel("τ (µs)", fontsize=8)
    fig.suptitle(f"g2: real vs Model A vs Model B (B scale {args.b_scale:.1e}), test set examples", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(OUT_DIR, "overview.png"), dpi=130)
    plt.close(fig)

    # every condition
    with PdfPages(os.path.join(OUT_DIR, "all_conditions.pdf")) as pdf:
        for k, i in enumerate(trials):
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
            draw_condition(axes, k, tau_us, ref, sim_a, sim_b, meta, i, r2a, r2b, True)
            for ax in axes:
                ax.set_xlabel("τ (µs)", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            if not args.no_png:
                fig.savefig(os.path.join(OUT_DIR, f"cond_{i:04d}.png"), dpi=110)
            plt.close(fig)
    print(f"saved {OUT_DIR}")


if __name__ == "__main__":
    main()
