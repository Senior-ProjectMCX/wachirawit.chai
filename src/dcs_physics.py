"""
DCS module: converts MCX photon pathlength/momentum-transfer data into g1(tau) / g2(tau).

Physics: Wu et al. (2021) Neurophotonics 8(1) 015001, Eq.(5) -- the momentum-transfer
Monte Carlo form (correct at short SDS), then the Siegert relation (Nakabayashi Eq. 2).
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

    using <dr^2(tau)>_i = 6*(alpha*D_B)_i*tau (Wu Eq.3), normalized so g1(0) = 1.

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


def synth_curves(det, fr_s, fr_d, beta, tau, trials, a, db0):
    """g2 of Model A for the given trials -> shape (3, len(tau), len(trials)).
    det[s] = (Ls, Ld, Ys, Yd) of SDS s; fr_s/fr_d indexed by trial; beta per SDS."""
    out = np.zeros((3, len(tau), len(trials)))
    for j, i in enumerate(trials):
        aDB_s = cfg.flow_to_alpha_db(fr_s[i], a, db0)
        aDB_d = cfg.flow_to_alpha_db(fr_d[i], a, db0)
        for s in range(3):
            Ls, Ld, Ys, Yd = det[s]
            out[s, :, j] = calculate_g2(Ls, Ld, Ys, Yd, aDB_s, aDB_d, tau, beta[s])
    return out
