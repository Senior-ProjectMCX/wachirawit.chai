"""
BFI module: fits a single Blood Flow Index (BFI = alpha*D_B) per (SDS, trial) from a
measured or simulated g2(tau) curve with the semi-infinite homogeneous model
(Nakabayashi Eq. 3). Used only for the BFI-X / BFI-Y metrics:

  BFI-X = BFI from this fitter; test: BFI-X(sim g2) / BFI-X(physical g2)
  BFI-Y = BFI column of metadata.csv (lab algorithm, unknown); test: BFI-X(sim g2) / BFI-Y
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg


def semi_infinite_g1(bfi, rho_mm, tau):
    """Normalized g1(tau) from the semi-infinite correlation-diffusion Green's function
    (Nakabayashi Eq.3 / Wu Eq.1-2). bfi = alpha*D_B in mm^2/s, rho in mm, tau in s."""
    r1 = np.sqrt(rho_mm ** 2 + cfg.Z0_MM ** 2)
    r2 = np.sqrt(rho_mm ** 2 + (cfg.Z0_MM + 2.0 * cfg.ZB_MM) ** 2)

    def _G1(k):
        return np.exp(-k * r1) / r1 - np.exp(-k * r2) / r2

    k_tau = np.sqrt(3.0 * cfg.MU_A_MM * cfg.MU_SP_MM
                    + (cfg.MU_SP_MM ** 2) * (cfg.K0 ** 2) * 6.0 * bfi * tau)
    k_static = np.sqrt(3.0 * cfg.MU_A_MM * cfg.MU_SP_MM)
    return _G1(k_tau) / _G1(k_static)


def fit_bfi_curve(g2_curve, rho_mm, beta, tau, bounds_log10=(-9.0, 1.0)):
    """Fit BFI (mm^2/s) and beta jointly to one g2(tau) curve, as Nakabayashi Eq. 3 does
    for each observation. BFI is fitted in log10; `beta` is the initial guess."""
    def resid(p):
        return 1.0 + p[1] * semi_infinite_g1(10.0 ** p[0], rho_mm, tau) ** 2 - g2_curve

    # Coarse 1-D search for a starting BFI with beta fixed, then refine both.
    start = minimize_scalar(lambda lb: np.mean(resid([lb, beta]) ** 2),
                            bounds=bounds_log10, method="bounded").x
    res = least_squares(resid, [start, beta],
                        bounds=([bounds_log10[0], 0.05], [bounds_log10[1], 1.0]))
    return 10.0 ** res.x[0]


def fit_bfi_all(g2, beta, tau):
    """g2 shape (3, N_TAU, n_trials) -> bfi shape (3, n_trials) in 1e-9 cm^2/s
    (the unit of the metadata BFI column)."""
    n_sds, _, n_trials = g2.shape
    bfi = np.zeros((n_sds, n_trials))
    for s in range(n_sds):
        for i in range(n_trials):
            bfi[s, i] = fit_bfi_curve(g2[s, :, i], cfg.SDS_MM[s], beta[s], tau)
    return bfi * cfg.BFI_MM2S_TO_FILE
