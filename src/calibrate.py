"""
Calibration of Model A (evaluation is in benchmark.py).

Calibration: find (a, D_B0) of alpha*D_B = a*max(FR_meas, 0) + D_B0 by least squares of
g2_sim vs g2_physical over all test-set trials and 3 SDS (tau every 5th point), minimized
with Nelder-Mead in log10 space (start 1e-6, 4.5e-7; bounds log10 a in [-9, -3],
log10 D_B0 in [-9, -5]; SciPy default stopping: xatol = fatol = 1e-4).
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg
from dcs_physics import synth_curves


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
