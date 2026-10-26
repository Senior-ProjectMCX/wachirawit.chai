"""
Run Model A and Model B on the SAME flow rate(s) and compare their g2 at SDS 1 / 2 / 3 cm.

  Model A  our MCX photons (paper geometry), alpha*D_B = a * max(FR, 0) + D_B0
           (a, D_B0 from plots/paper_geometry_compare.json, model "paper")
  Model B  Wichuda_r as her code runs it (../wichuda_g2): her photons, alpha*D_B = 0.4e-6 * FR
           (Wichuda_r commit 64985e7), no D_B0

Both models get exactly the flow you pass. beta:
  --beta same (default)  both use beta per SDS = mean real g2(0) - 1, so only the model differs
  --beta own             each model's own convention: A per SDS; B per condition fitted from the real
                         trials of that preset condition (falls back to per SDS if the condition is
                         not in the dataset)

If the flow pair matches a preset condition in the dataset, the mean real g2 of that condition is
overlaid and R^2 (1 - SS_res/SS_tot) of each model against it is printed.

Usage:
  .venv/bin/python scripts/run_two_models.py --flows 5,20
  .venv/bin/python scripts/run_two_models.py --flows 0,5 10,10 70,70 --beta own
Output: plots/two_models/two_models_<shallow>_<deep>.png and .csv (g2 of both models, and real if any)
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "wichuda_g2"))
from config import parameters as cfg
import bfi_fit
import data_io
from compare import _r2, BFI_MM2S_TO_FILE
from dcs_physics import calculate_g2
import wichuda_g2 as wg

B_SCALE = 0.4e-6                      # Wichuda_r commit 64985e7
OUT_DIR = os.path.join(ROOT, "plots", "two_models")
SDS = ["1 cm", "2 cm", "3 cm"]


def load_model_a():
    path = os.path.join(ROOT, "plots", "paper_geometry_compare.json")
    cal = json.load(open(path))["models"]["paper"]
    pl = np.load(os.path.join(ROOT, "data", "pathlengths_paper_geom.npz"))
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
    return cal["a"], cal["db0"], det


def model_a_g2(fs, fd, a, db0, det, beta, tau):
    aDB_s = cfg.flow_to_alpha_db(fs, a=a, db0=db0)
    aDB_d = cfg.flow_to_alpha_db(fd, a=a, db0=db0)
    g2 = np.array([calculate_g2(*det[s], aDB_s, aDB_d, tau, beta[s]) for s in range(3)])
    return g2, aDB_s, aDB_d


def model_b_g2(fs, fd, beta, detp, tau):
    wg.flow_to_alpha_db = lambda f: max(float(f), 0.0) * B_SCALE
    return wg.generate_g2(fs, fd, beta, detp=detp, tau_seconds=tau), fs * B_SCALE, fd * B_SCALE


def main():
    ap = argparse.ArgumentParser(description="Run Model A and Model B on the same flow rate(s)")
    ap.add_argument("--flows", nargs="+", required=True,
                    help="one or more 'shallow,deep' pairs in ml/min/100g, e.g. 5,20 10,10")
    ap.add_argument("--beta", choices=["same", "own"], default="same")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = data_io.load_physical_metadata()
    g2_real_all = data_io.load_physical_g2()
    beta_sds = data_io.extract_beta(g2_real_all)
    tau = data_io.TAU_ARRAY
    tu = tau * 1e6
    a, db0, det = load_model_a()
    detp = wg.load_detp()
    print(f"Model A: alpha*D_B = {a:.3e} * FR + {db0:.3e} mm^2/s | Model B: alpha*D_B = {B_SCALE:.1e} * FR")
    print(f"beta mode: {args.beta}")

    for pair in args.flows:
        fs, fd = (float(x) for x in pair.split(","))
        same = np.where((meta["fr_set_s"] == fs) & (meta["fr_set_d"] == fd))[0]
        real = g2_real_all[:, :, same].mean(axis=2) if len(same) else None

        beta_b = beta_sds
        if args.beta == "own" and len(same):
            beta_b = np.array([wg.beta_from_physical_trials(g2_real_all[s][:, same].T) for s in range(3)])

        g2a, aDa_s, aDa_d = model_a_g2(fs, fd, a, db0, det, beta_sds, tau)
        g2b, aDb_s, aDb_d = model_b_g2(fs, fd, beta_b, detp, tau)
        bfi_a = bfi_fit.fit_bfi_all(g2a[:, :, None], beta_sds, tau)[:, 0] * BFI_MM2S_TO_FILE
        bfi_b = bfi_fit.fit_bfi_all(g2b[:, :, None], beta_b, tau)[:, 0] * BFI_MM2S_TO_FILE
        bfi_r = (bfi_fit.fit_bfi_all(real[:, :, None], beta_sds, tau)[:, 0] * BFI_MM2S_TO_FILE
                 if real is not None else None)

        print(f"\n== flow (shallow, deep) = ({fs:g}, {fd:g}) ml/min/100g ==")
        print(f"  alpha*D_B shallow / deep  A {aDa_s:.3e} / {aDa_d:.3e}   B {aDb_s:.3e} / {aDb_d:.3e} mm^2/s")
        print("  real data: " + (f"{len(same)} trials of this preset condition (mean shown)" if real is not None
                                  else "no preset condition with this flow in the dataset"))
        for s in range(3):
            line = f"  SDS {SDS[s]}: BFI-X A {bfi_a[s]:7.2f}  B {bfi_b[s]:7.2f}"
            if real is not None:
                line += (f"  real {bfi_r[s]:7.2f} | R2 A {_r2(g2a[s], real[s]):.3f}  "
                         f"B {_r2(g2b[s], real[s]):.3f}")
            print(line)

        m = tu > 0
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for s, ax in enumerate(axes):
            if real is not None:
                ax.semilogx(tu[m], real[s, m], color="black", lw=2.2, label=f"real (mean of {len(same)})")
            ax.semilogx(tu[m], g2a[s, m], color="#2a78d6", lw=1.8, label="Model A")
            ax.semilogx(tu[m], g2b[s, m], color="#eb6834", lw=1.8, ls="--", label="Model B")
            title = f"SDS {SDS[s]}"
            if real is not None:
                title += f"   R² A {_r2(g2a[s], real[s]):.2f} | B {_r2(g2b[s], real[s]):.2f}"
            ax.set_title(title, fontsize=9)
            ax.set_xlim(1, 500)
            ax.set_xlabel("τ (µs)", fontsize=8)
            ax.grid(alpha=0.3, which="both", lw=0.4)
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel("g2", fontsize=8)
        axes[0].legend(fontsize=7, frameon=False, loc="lower left")
        fig.suptitle(f"flow (shallow, deep) = ({fs:g}, {fd:g}) ml/min/100g", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        stem = os.path.join(OUT_DIR, f"two_models_{fs:g}_{fd:g}")
        fig.savefig(stem + ".png", dpi=120)
        plt.close(fig)

        cols = {"tau_us": tu}
        for s in range(3):
            cols[f"g2_A_SDS{s + 1}"] = g2a[s]
            cols[f"g2_B_SDS{s + 1}"] = g2b[s]
            if real is not None:
                cols[f"g2_real_SDS{s + 1}"] = real[s]
        pd.DataFrame(cols).to_csv(stem + ".csv", index=False)
        print(f"  saved {stem}.png / .csv")


if __name__ == "__main__":
    main()
