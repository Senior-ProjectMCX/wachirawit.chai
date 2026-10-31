"""
Figures of Model A: g2 autocorrelation of every flow condition, real (black, solid) vs
Model A (blue, dashed). Real and Model A curves are the mean over the condition's trials.

Output (final_result/):
  conditions/cond_<shallow>_<deep>.png   one figure per condition, SDS 1 / 2 / 3 cm side by side
  all_conditions.pdf                     the same figures, one per page
  grid_SDS1cm.png, grid_SDS2cm.png, grid_SDS3cm.png   all conditions as small multiples per SDS
  metrics_per_condition.csv              R^2 and RMSE of Model A against the condition-mean real curve
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import r2

SDS = ["1 cm", "2 cm", "3 cm"]
C_REAL, C_A = "black", "#2a78d6"
LS_A = (0, (5, 2))


def _draw(ax, tu, real, model, small=False):
    ax.semilogx(tu, real, color=C_REAL, lw=1.4 if small else 2.2, label="Real (mean of trials)")
    ax.semilogx(tu, model, color=C_A, lw=1.1 if small else 1.8, ls=LS_A, label="Model A")
    ax.set_xlim(1, 500)
    ax.grid(alpha=0.3, which="both", lw=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_all(out_dir, tau, meta, g2, sim, trials):
    """sim shape (3, N_TAU, len(trials)), columns aligned with `trials`."""
    os.makedirs(os.path.join(out_dir, "conditions"), exist_ok=True)
    m = tau > 0
    tu = tau[m] * 1e6
    conds = sorted({(meta["fr_set_s"][i], meta["fr_set_d"][i]) for i in trials})
    real_c, sim_c, rows = {}, {}, []
    for c in conds:
        idx = np.where((meta["fr_set_s"][trials] == c[0]) & (meta["fr_set_d"][trials] == c[1]))[0]
        real_c[c] = g2[:, :, trials[idx]].mean(axis=2)
        sim_c[c] = sim[:, :, idx].mean(axis=2)
        row = {"FR_set_shallow": c[0], "FR_set_deep": c[1], "n_trials": int(len(idx))}
        for s in range(3):
            row[f"R2_SDS{s + 1}cm"] = r2(sim_c[c][s], real_c[c][s])
            row[f"RMSE_SDS{s + 1}cm"] = float(np.sqrt(np.mean((sim_c[c][s] - real_c[c][s]) ** 2)))
        rows.append(row)
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(out_dir, "metrics_per_condition.csv"), index=False)

    with PdfPages(os.path.join(out_dir, "all_conditions.pdf")) as pdf:
        for c in conds:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            for s, ax in enumerate(axes):
                _draw(ax, tu, real_c[c][s][m], sim_c[c][s][m])
                ax.set_title(f"SDS {SDS[s]}   R² {r2(sim_c[c][s], real_c[c][s]):.3f}", fontsize=9)
                ax.set_xlabel("τ (µs)", fontsize=8)
                ax.tick_params(labelsize=7)
            axes[0].set_ylabel("g2(τ)", fontsize=9)
            axes[0].legend(fontsize=7, frameon=False, loc="lower left")
            fig.suptitle(f"Flow (shallow, deep) = ({c[0]:g}, {c[1]:g}) ml/min/100g", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(out_dir, "conditions", f"cond_{c[0]:g}_{c[1]:g}.png"), dpi=120)
            pdf.savefig(fig)
            plt.close(fig)

    ncol = 12
    nrow = int(np.ceil(len(conds) / ncol))
    for s in range(3):
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 1.55), sharex=True)
        for k, ax in enumerate(axes.flat):
            if k >= len(conds):
                ax.axis("off")
                continue
            c = conds[k]
            _draw(ax, tu, real_c[c][s][m], sim_c[c][s][m], small=True)
            ax.set_title(f"({c[0]:g}, {c[1]:g})", fontsize=7, pad=2)
            ax.tick_params(labelsize=5, length=2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, frameon=False,
                   bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"g2(τ) at SDS {SDS[s]} for all {len(conds)} flow conditions (shallow, deep) ml/min/100g; "
                     f"x: τ 1–500 µs (log), y: g2", fontsize=11, y=0.975)
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        fig.savefig(os.path.join(out_dir, f"grid_SDS{s + 1}cm.png"), dpi=110)
        plt.close(fig)
    return met
