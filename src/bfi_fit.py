"""
BFI module: fits a single Blood Flow Index (BFI = alpha*D_B) per (SDS, trial)
from a measured/simulated g2(tau) curve, using the semi-infinite homogeneous
medium Green's function model -- the same model the DCS hardware/paper uses,
which is why it CANNOT separate shallow vs deep flow on its own (that's the
LSTM's job downstream).

Also provides the c2/c1 geometry-validation gate: BFI = c1*FR_s + c2*FR_d + c3
(Nakabayashi Eq. 10-12). The ratio c2/c1 depends only on geometry/optics, not
on the flow->D_B calibration scale, so it must match the paper's target BEFORE
any calibration is trusted (see core/code/c2c1_test.py, ported here unchanged
except for pulling constants from config.parameters).
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg


def semi_infinite_g1(bfi, rho_mm, tau):
    """
    Normalized g1(tau) from the semi-infinite correlation-diffusion Green's
    function (Nakabayashi Eq.3 / Wu Eq.1-2). bfi = alpha*D_B in mm^2/s,
    rho = source-detector separation in mm, tau in seconds.
    """
    r1 = np.sqrt(rho_mm ** 2 + cfg.Z0_MM ** 2)
    r2 = np.sqrt(rho_mm ** 2 + (cfg.Z0_MM + 2.0 * cfg.ZB_MM) ** 2)

    def _G1(k):
        return np.exp(-k * r1) / r1 - np.exp(-k * r2) / r2

    k_tau = np.sqrt(3.0 * cfg.MU_A_MM * cfg.MU_SP_MM
                     + (cfg.MU_SP_MM ** 2) * (cfg.K0 ** 2) * 6.0 * bfi * tau)
    k_static = np.sqrt(3.0 * cfg.MU_A_MM * cfg.MU_SP_MM)
    return _G1(k_tau) / _G1(k_static)


def fit_bfi_curve(g2_curve, rho_mm, beta, tau, bounds_log10=(-9.0, 1.0)):
    """Least-squares fit of a single BFI (mm^2/s) to one g2(tau) curve, in
    log10(BFI) space since BFI spans orders of magnitude. beta is fixed."""
    def loss(log_bfi):
        g1 = semi_infinite_g1(10.0 ** log_bfi, rho_mm, tau)
        g2_model = 1.0 + beta * (g1 ** 2)
        return np.mean((g2_model - g2_curve) ** 2)

    res = minimize_scalar(loss, bounds=bounds_log10, method="bounded")
    return 10.0 ** res.x


def fit_bfi_all(g2, beta, tau):
    """g2 shape (3, N_TAU, n_trials) -> bfi shape (3, n_trials)."""
    n_sds, _, n_trials = g2.shape
    bfi = np.zeros((n_sds, n_trials))
    for s in range(n_sds):
        for i in range(n_trials):
            bfi[s, i] = fit_bfi_curve(g2[s, :, i], cfg.SDS_MM[s], beta[s], tau)
    return bfi


def regress_c2c1(bfi, fr_s, fr_d):
    """Per-SDS OLS fit BFI = c1*FR_s + c2*FR_d + c3. Returns list of
    (c1, c2, c3, r2, c2/c1), one tuple per SDS. bfi shape (3, n_trials)."""
    A = np.column_stack([fr_s, fr_d, np.ones_like(fr_s)])
    out = []
    for s in range(bfi.shape[0]):
        y = bfi[s]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        c1, c2, c3 = coef
        resid = y - A @ coef
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ratio = c2 / c1 if c1 != 0 else float("nan")
        out.append((c1, c2, c3, r2, ratio))
    return out


def geometry_gate_passed(results, tolerance=0.06, target=None):
    """True if every SDS's c2/c1 is within `tolerance` of the target ratio."""
    target = target or cfg.TARGET_C2_C1
    return all(abs(results[s][4] - target[s]) <= tolerance for s in range(len(results)))


def print_regression(title, results, reference=None, ref_label="target"):
    ref = reference if reference is not None else cfg.TARGET_C2_C1
    print(f"\n{title}")
    print(f"  {'SDS':<5} {'c1':>11} {'c2':>11} {'c3':>11} {'R2':>7} {'c2/c1':>9} {ref_label:>9} {'status':>8}")
    for s, (c1, c2, c3, r2, ratio) in enumerate(results):
        tgt = ref[s]
        ok = "PASS" if abs(ratio - tgt) <= 0.06 else "off"
        print(f"  {cfg.SDS_LABELS[s]:<5} {c1:>11.3e} {c2:>11.3e} {c3:>11.3e} "
              f"{r2:>7.3f} {ratio:>9.3f} {tgt:>9.3f} {ok:>8}")
