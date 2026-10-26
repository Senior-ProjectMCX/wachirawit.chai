"""
Zero flow: ideal (no motion at all, alpha*D_B = 0 -> g1 = 1, g2 = 1 + beta, a flat line) vs the real
phantom g2 at the (0,0) condition (both pumps off, 210 trials) vs Model A at FR = 0
(alpha*D_B = D_B0, Brownian baseline). Model B at FR = 0 has alpha*D_B = 0, i.e. the ideal line.

  beta per SDS = mean real g2(0) - 1 over all trials (same as the models)
  Model A: paper-geometry photons, D_B0 from plots/paper_geometry_compare.json
Usage: .venv/bin/python scripts/plot_zero_flow_ideal.py [SDS index 1|2|3, default 2]
Output: plots/zero_flow_ideal_vs_real.png
"""
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
import data_io
from compare import _synth_curves

meta = data_io.load_physical_metadata()
g2 = data_io.load_physical_g2()
beta = data_io.extract_beta(g2)
tau = data_io.TAU_ARRAY
tu = tau * 1e6
zero = np.where(data_io.excluded_condition(meta["fr_set_s"], meta["fr_set_d"]))[0]

cal = json.load(open(os.path.join(ROOT, "plots", "paper_geometry_compare.json")))["models"]["paper"]
pl = np.load(os.path.join(ROOT, "data", "pathlengths_paper_geom.npz"))
rng = np.random.default_rng(42)
det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
zeros = np.zeros(1)
sim_a = _synth_curves(det, zeros, zeros, beta, tau, np.array([0]), cal["a"], cal["db0"])[:, :, 0]

m = tu > 0
sds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
fig, ax = plt.subplots(figsize=(7, 4.4))
for s in [sds - 1]:
    real = g2[s][:, zero]                       # (501, n)
    for j in range(real.shape[1]):
        ax.semilogx(tu[m], real[m, j], color="0.75", lw=0.4, alpha=0.35,
                    label="real (0,0): each trial" if j == 0 else None)
    ax.semilogx(tu[m], real[m].mean(axis=1), color="black", lw=2.2, label=f"real (0,0): mean of {real.shape[1]} trials")
    ax.semilogx(tu[m], np.full(m.sum(), 1 + beta[s]), color="#eb6834", lw=2, ls="--",
                label="ideal: no motion (αD_B = 0) = Model B at flow 0")
    ax.set_xlim(1, 500)
    ax.set_xlabel("τ (µs)", fontsize=9)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
ax.set_ylabel("g2", fontsize=9)
ax.legend(fontsize=8, loc="lower left", frameon=False)
ax.set_title(f"Zero flow, SDS {sds} cm: ideal (no motion) vs real phantom", fontsize=10)
fig.tight_layout()
out = os.path.join(ROOT, "plots", "zero_flow_ideal_vs_real.png")
fig.savefig(out, dpi=130)
print(f"saved {out}  ({len(zero)} real trials)")
for s in range(3):
    r = g2[s][:, zero].mean(axis=1)
    print(f"SDS {s+1}: real g2 at 1us {r[1]:.3f} -> 500us {r[500]:.3f} | ideal {1+beta[s]:.3f} flat | "
          f"Model A {sim_a[s,1]:.3f} -> {sim_a[s,500]:.3f}")
