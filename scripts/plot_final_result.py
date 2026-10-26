"""
Final-setting figures: g2 autocorrelation of every flow condition, real vs Model A vs Model B.

Final setting
  Data       K = 5 closest trials per condition (dataset/testset_closest_k5.csv, 132 conditions,
             (0,0) excluded). Real curve = mean of the 5 trials of the condition.
  Model A    paper-geometry MCX, 1e8 photons (data/pathlengths_paper_geom_n1e08.npz), 5,000 photons
             per SDS; alpha*D_B = a*max(FR_meas, 0) + D_B0 with (a, D_B0) fitted on the 660 trials
             (Nelder-Mead, log10, least squares of g2, tau every 5th point). Per-trial g2 from each
             trial's measured flow, averaged over the 5 trials like the real curve.
  Model B    Wichuda_r (commit 64985e7) with the pathlength unit fixed (voxel -> mm, x0.5 on a loaded
             copy): alpha*D_B = 0.4e-6 * preset flow, beta per condition.

Output (final_result/):
  conditions/cond_<shallow>_<deep>.png   one figure per condition, SDS 1 / 2 / 3 cm side by side
  all_conditions.pdf                     the same 132 figures, one per page
  grid_SDS1cm.png, grid_SDS2cm.png, grid_SDS3cm.png   all 132 conditions as small multiples per SDS
  metrics_per_condition.csv              R^2 and RMSE of A and B against the condition-mean real curve
  settings.json                          the fitted (a, D_B0) and file paths used

Usage: .venv/bin/python scripts/plot_final_result.py
"""
import json
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
import data_io
from compare import _synth_curves, _r2
from k_test_fixed_b import calibrate, run_b, VOXEL_MM
import wichuda_g2 as wg

OUT = os.path.join(ROOT, "final_result")
PHOTONS = os.path.join(ROOT, "data", "pathlengths_paper_geom_n1e08.npz")
TESTSET = os.path.join(ROOT, "dataset", "testset_closest_k5.csv")
SDS = ["1 cm", "2 cm", "3 cm"]
C_REAL, C_A, C_B = "black", "#2a78d6", "#e8590c"
LS_A, LS_B = (0, (5, 2)), (0, (2, 1.5))


def draw(ax, tu, m, real, a, b, small=False):
    ax.semilogx(tu[m], real[m], color=C_REAL, lw=1.4 if small else 2.2, label="Real (mean of 5 trials)")
    ax.semilogx(tu[m], a[m], color=C_A, lw=1.1 if small else 1.8, ls=LS_A, label="Model A")
    ax.semilogx(tu[m], b[m], color=C_B, lw=1.1 if small else 1.8, ls=LS_B, label="Model B")
    ax.set_xlim(1, 500)
    ax.grid(alpha=0.3, which="both", lw=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    os.makedirs(os.path.join(OUT, "conditions"), exist_ok=True)
    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    tu = tau * 1e6
    m = tu > 0

    ts = pd.read_csv(TESTSET)
    trials = np.sort(ts["Trial_Index"].to_numpy(int))
    pl = np.load(PHOTONS)
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
    a, db0, res = calibrate(det, meta, beta, tau, g2, trials)
    print(f"Model A (1e8 photons) fitted on {len(trials)} trials: a = {a:.4e}, D_B0 = {db0:.4e} "
          f"({res.nfev} evals, RMSE {np.sqrt(res.fun):.4f})", flush=True)

    sim_a = _synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)
    detp = wg.load_detp().astype(float)
    detp[1:3] *= VOXEL_MM
    sim_b = run_b(detp, meta, g2, trials, tau)

    conds = sorted({(meta["fr_set_s"][i], meta["fr_set_d"][i]) for i in trials})
    real_c, a_c, b_c, rows = {}, {}, {}, []
    for c in conds:
        idx = np.where((meta["fr_set_s"][trials] == c[0]) & (meta["fr_set_d"][trials] == c[1]))[0]
        real_c[c] = g2[:, :, trials[idx]].mean(axis=2)
        a_c[c] = sim_a[:, :, idx].mean(axis=2)
        b_c[c] = sim_b[:, :, idx].mean(axis=2)
        row = {"FR_set_shallow": c[0], "FR_set_deep": c[1], "n_trials": int(len(idx))}
        for s in range(3):
            lab = f"SDS{s + 1}cm"
            row[f"R2_A_{lab}"] = _r2(a_c[c][s], real_c[c][s])
            row[f"R2_B_{lab}"] = _r2(b_c[c][s], real_c[c][s])
            row[f"RMSE_A_{lab}"] = float(np.sqrt(np.mean((a_c[c][s] - real_c[c][s]) ** 2)))
            row[f"RMSE_B_{lab}"] = float(np.sqrt(np.mean((b_c[c][s] - real_c[c][s]) ** 2)))
        rows.append(row)
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(OUT, "metrics_per_condition.csv"), index=False)

    # one figure per condition + combined PDF
    with PdfPages(os.path.join(OUT, "all_conditions.pdf")) as pdf:
        for c in conds:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            for s, ax in enumerate(axes):
                draw(ax, tu, m, real_c[c][s], a_c[c][s], b_c[c][s])
                ax.set_title(f"SDS {SDS[s]}   R² A {_r2(a_c[c][s], real_c[c][s]):.3f} | "
                             f"B {_r2(b_c[c][s], real_c[c][s]):.3f}", fontsize=9)
                ax.set_xlabel("τ (µs)", fontsize=8)
                ax.tick_params(labelsize=7)
            axes[0].set_ylabel("g2(τ)", fontsize=9)
            axes[0].legend(fontsize=7, frameon=False, loc="lower left")
            fig.suptitle(f"Flow (shallow, deep) = ({c[0]:g}, {c[1]:g}) ml/min/100g", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(OUT, "conditions", f"cond_{c[0]:g}_{c[1]:g}.png"), dpi=120)
            pdf.savefig(fig)
            plt.close(fig)

    # small multiples: all conditions, one figure per SDS
    ncol = 12
    nrow = int(np.ceil(len(conds) / ncol))
    for s in range(3):
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 1.55), sharex=True)
        for k, ax in enumerate(axes.flat):
            if k >= len(conds):
                ax.axis("off")
                continue
            c = conds[k]
            draw(ax, tu, m, real_c[c][s], a_c[c][s], b_c[c][s], small=True)
            ax.set_title(f"({c[0]:g}, {c[1]:g})", fontsize=7, pad=2)
            ax.tick_params(labelsize=5, length=2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, frameon=False,
                   bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"g2(τ) at SDS {SDS[s]} for all {len(conds)} flow conditions (shallow, deep) ml/min/100g; "
                     f"x: τ 1–500 µs (log), y: g2", fontsize=11, y=0.975)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        fig.savefig(os.path.join(OUT, f"grid_SDS{s + 1}cm.png"), dpi=110)
        plt.close(fig)

    settings = {"model_A": {"photons_file": os.path.relpath(PHOTONS, ROOT), "photons_launched": 1e8,
                            "a_mm2_s_per_flow": a, "db0_mm2_s": db0, "fit_trials": int(len(trials))},
                "model_B": {"source": "Wichuda_r commit 64985e7", "flow_to_alphaDB": "0.4e-6 * preset flow",
                            "pathlength_unit_fix": "x0.5 (voxel -> mm)", "beta": "per condition"},
                "data": {"testset": os.path.relpath(TESTSET, ROOT), "k": 5, "conditions": len(conds),
                         "real_curve": "mean of the 5 trials per condition"}}
    with open(os.path.join(OUT, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    for s in range(3):
        lab = f"SDS{s + 1}cm"
        print(f"{SDS[s]}: condition-mean R2 median A {met[f'R2_A_{lab}'].median():.3f} B {met[f'R2_B_{lab}'].median():.3f}"
              f" | A better in {int((met[f'R2_A_{lab}'] > met[f'R2_B_{lab}']).sum())}/{len(met)}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
