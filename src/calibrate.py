"""
Calibration + evaluation of Model A.

Calibration: find (a, D_B0) of alpha*D_B = a*max(FR_meas, 0) + D_B0 by least squares of
g2_sim vs g2_physical over all test-set trials and 3 SDS (tau every 5th point), minimized
with Nelder-Mead in log10 space (start 1e-6, 4.5e-7; bounds log10 a in [-9, -3],
log10 D_B0 in [-9, -5]; SciPy default stopping: xatol = fatol = 1e-4).

Evaluation (same trials): per-curve R^2 = 1 - SS_res/SS_tot (median, mean), pooled RMSE of
g2, and BFI ratios sim / reference (median, mean) for BFI-X and BFI-Y (see bfi_fit.py).
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg
import bfi_fit
from dcs_physics import synth_curves


def r2(sim, ref):
    """R^2 = 1 - SS_res/SS_tot over all points of the given arrays."""
    ss_res = np.sum((ref - sim) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def calibrate(det, meta, beta, tau, g2, trials):
    """Nelder-Mead fit of (a, D_B0). Returns a, D_B0 and the SciPy result."""
    tc = tau[::cfg.CAL_TAU_STRIDE]
    ref = g2[:, ::cfg.CAL_TAU_STRIDE, :][:, :, trials]

    def loss(p):
        sim = synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tc, trials,
                           10.0 ** p[0], 10.0 ** p[1])
        return np.mean((sim - ref) ** 2)

    x0 = [np.log10(cfg.START_A_MM2_S_PER_FR), np.log10(cfg.START_DB0_MM2_S)]
    res = minimize(loss, x0, method="Nelder-Mead", bounds=cfg.CAL_BOUNDS_LOG10)
    return 10.0 ** res.x[0], 10.0 ** res.x[1], res


def evaluate(sim, ref, beta, tau, bfi_y):
    """Metrics per SDS. sim/ref shape (3, N_TAU, n_trials); bfi_y shape (3, n_trials)."""
    n = ref.shape[2]
    r2_all = np.array([[r2(sim[s, :, k], ref[s, :, k]) for k in range(n)] for s in range(3)])
    bfi_sim = bfi_fit.fit_bfi_all(sim, beta, tau)
    bfi_real = bfi_fit.fit_bfi_all(ref, beta, tau)
    rx, ry = bfi_sim / bfi_real, bfi_sim / bfi_y
    return {
        "r2_median": [float(np.median(r2_all[s])) for s in range(3)],
        "r2_mean": [float(np.mean(r2_all[s])) for s in range(3)],
        "rmse_pooled": [float(np.sqrt(np.mean((sim[s] - ref[s]) ** 2))) for s in range(3)],
        "bfiX_ratio_median": [float(np.median(rx[s])) for s in range(3)],
        "bfiX_ratio_mean": [float(np.mean(rx[s])) for s in range(3)],
        "bfiY_ratio_median": [float(np.median(ry[s])) for s in range(3)],
        "bfiY_ratio_mean": [float(np.mean(ry[s])) for s in range(3)],
    }
