"""
DCS module: converts MCX photon pathlength/momentum-transfer data into
g1(tau) / g2(tau) autocorrelation curves for a given flow condition.

Physics: Wu et al. (2021) Neurophotonics 8(1) 015001, Eq.(5) -- the
momentum-transfer Monte Carlo form (correct at short SDS), NOT the diffusion
approximation mu'_s * L used in the first core/ prototype (bug 3.4/3.5 in
the analysis doc). Ported from core/code/dynamic_flow.py; the only physics
change here is that flow->alpha*Db goes through config.parameters.flow_to_alpha_db,
which adds a thermal-Brownian baseline D_B0 so g2 does not go flat at FR=0
(bug 3.7).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg


def calculate_g1(Ls, Ld, Ys, Yd, alpha_db_s, alpha_db_d, tau_array,
                  mua_s=cfg.MU_A_MM, mua_d=cfg.MU_A_MM):
    """
    Normalized electric-field autocorrelation g1(tau):

        G1(tau) = (1/Np) * sum_n exp(-2 k0^2 (aD_s Y_{n,s} + aD_d Y_{n,d}) tau)
                                * exp(-(mua_s L_{n,s} + mua_d L_{n,d}))

    using <dr^2(tau)>_i = 6*(alpha*D_B)_i*tau (Wu Eq.3). g1 is normalized by
    the static (tau=0) absorption-weighted photon count, so g1(0) = 1.

    Parameters
    ----------
    Ls, Ld : per-photon partial pathlength in shallow / deep layer (mm)
    Ys, Yd : per-photon momentum transfer sum(1-cos theta) in shallow / deep layer
    alpha_db_s, alpha_db_d : alpha*D_B (mm^2/s) for the shallow / deep layer
    tau_array : delay times (seconds)
    """
    if len(Ls) == 0:
        return np.ones_like(tau_array)

    abs_w = np.exp(-(mua_s * Ls + mua_d * Ld))          # (Np,) static absorption weight
    decay_rates = 2.0 * (cfg.K0 ** 2) * (alpha_db_s * Ys + alpha_db_d * Yd)  # (Np,) s^-1

    norm = np.sum(abs_w)
    if norm == 0:
        return np.ones_like(tau_array)

    g1 = np.empty(len(tau_array))
    for t_idx, tau in enumerate(tau_array):
        g1[t_idx] = np.sum(abs_w * np.exp(-decay_rates * tau)) / norm
    return g1


def calculate_g2(Ls, Ld, Ys, Yd, alpha_db_s, alpha_db_d, tau_array, beta,
                  mua_s=cfg.MU_A_MM, mua_d=cfg.MU_A_MM):
    """Intensity autocorrelation via the Siegert relation: g2 = 1 + beta*|g1|^2."""
    g1 = calculate_g1(Ls, Ld, Ys, Yd, alpha_db_s, alpha_db_d, tau_array,
                       mua_s=mua_s, mua_d=mua_d)
    return 1.0 + beta * (g1 ** 2)


def g2_for_condition(pathlengths, fr_shallow, fr_deep, beta, tau_array=None,
                      a=cfg.DEFAULT_A_MM2_S_PER_FR, db0=cfg.DEFAULT_DB0_MM2_S):
    """
    Convenience wrapper: one flow condition -> g2(tau) for all 3 SDS.

    pathlengths : dict with keys det_{1,2,3}_{Ls,Ld,Ys,Yd} (see mcx_engine.run_simulation)
    Returns array shape (3, len(tau_array)).
    """
    tau_array = cfg.TAU_S if tau_array is None else tau_array
    aDB_s = cfg.flow_to_alpha_db(fr_shallow, a=a, db0=db0)
    aDB_d = cfg.flow_to_alpha_db(fr_deep, a=a, db0=db0)

    out = np.zeros((len(cfg.SDS_LABELS), len(tau_array)))
    for s in range(len(cfg.SDS_LABELS)):
        det_idx = s + 1
        Ls = pathlengths[f"det_{det_idx}_Ls"]
        Ld = pathlengths[f"det_{det_idx}_Ld"]
        Ys = pathlengths[f"det_{det_idx}_Ys"]
        Yd = pathlengths[f"det_{det_idx}_Yd"]
        beta_s = beta[s] if hasattr(beta, "__len__") else beta
        out[s] = calculate_g2(Ls, Ld, Ys, Yd, aDB_s, aDB_d, tau_array, beta_s)
    return out
