"""
Whole-model BFI overview, Model A vs Model B, on the 132-condition test set.
  Row 1  BFI-X(sim) vs BFI-X(real g2), log-log, per SDS, with y = x and x2 / x0.5 bands
  Row 2  BFI-X(sim) vs BFI-Y (metadata.csv column, unknown algorithm), same layout
  Row 3  distribution of the per-trial ratio sim / reference (log scale), BFI-X and BFI-Y
Input: plots/g2_overlay_testset_paper_geom/r2_per_curve.csv (BFI-X real / A / B per trial,
       written by scripts/plot_g2_overlay.py) + metadata.csv for BFI-Y.
Usage: .venv/bin/python scripts/plot_bfi_overview.py
Output: plots/bfi_overview_paper_geom.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data_io

IN = os.path.join(ROOT, "plots", "g2_overlay_testset_paper_geom", "r2_per_curve.csv")
OUT = os.path.join(ROOT, "plots", "bfi_overview_paper_geom.png")
SDS = ["1cm", "2cm", "3cm"]
COL = {"A": "tab:blue", "B": "tab:red"}


def main():
    d = pd.read_csv(IN)
    meta = data_io.load_physical_metadata()
    t = d["Trial_Index"].to_numpy()

    fig, axes = plt.subplots(3, 3, figsize=(13, 12.5))
    for s, lab in enumerate(SDS):
        refs = {"BFI-X": d[f"BFI_real_{lab}"].to_numpy(), "BFI-Y": meta["bfi"][s, t]}
        for row, (name, ref) in enumerate(refs.items()):
            ax = axes[row, s]
            lo, hi = 0.1, 1000
            xx = np.array([lo, hi])
            ax.fill_between(xx, xx * 0.5, xx * 2, color="0.9", zorder=0, label="×0.5 – ×2")
            ax.plot(xx, xx, color="black", lw=1, label="y = x")
            for m in ("A", "B"):
                sim = d[f"BFI_{m}_{lab}"].to_numpy()
                r = np.corrcoef(sim, ref)[0, 1]
                ratio = np.median(sim / ref)
                ax.scatter(ref, sim, s=12, alpha=0.75, color=COL[m],
                           label=f"Model {m}: R {r:.3f}, ratio {ratio:.2f}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{name} reference (1e-9 cm²/s)", fontsize=8)
            ax.set_ylabel("BFI-X of sim (1e-9 cm²/s)", fontsize=8)
            ax.set_title(f"SDS {lab[0]} cm — sim vs {name}", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
            ax.tick_params(labelsize=7)

        ax = axes[2, s]
        data, labels, colors = [], [], []
        for name, ref in (("X", d[f"BFI_real_{lab}"].to_numpy()), ("Y", meta["bfi"][s, t])):
            for m in ("A", "B"):
                data.append(d[f"BFI_{m}_{lab}"].to_numpy() / ref)
                labels.append(f"{m}\nBFI-{name}")
                colors.append(COL[m])
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                        flierprops={"markersize": 3})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        ax.axhline(1, color="black", lw=1)
        ax.axhspan(0.5, 2, color="0.9", zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 30)
        ax.set_ylabel("ratio sim ÷ reference (per trial)", fontsize=8)
        ax.set_title(f"SDS {lab[0]} cm — ratio distribution (132 trials)", fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle("BFI overview: Model A vs Model B, 132 conditions "
                 "(BFI-X = our fitter; BFI-Y = metadata, unknown algorithm)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
