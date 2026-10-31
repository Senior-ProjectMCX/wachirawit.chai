"""
Generate Model A g2(tau) for any (shallow, deep) flow rate from the photon data already
simulated (data/pathlengths.npz) and the fitted model (final_result/model_a.json: a, D_B0,
beta per SDS). No MCX run is needed; re-run MCX only if geometry or optics change.

    from generate import generate_g2
    g2 = generate_g2([(5, 20), (12, 30)])      # shape (2, 3, 501): flow, SDS 1/2/3 cm, tau 0..500 us
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg
import data_io
from dcs_physics import calculate_g2

MODEL_PATH = os.path.join(data_io.ROOT, "final_result", "model_a.json")


def load_model(path=MODEL_PATH):
    """(a, D_B0, beta[3]) from the calibration output."""
    with open(path) as f:
        m = json.load(f)
    return m["a_mm2_s_per_flow"], m["db0_mm2_s"], np.array(m["beta"])


def generate_g2(flows, pathlengths=data_io.PATHLENGTHS_PATH, model=MODEL_PATH, tau=data_io.TAU_ARRAY,
                a=None, db0=None, beta=None):
    """
    flows : list of (shallow, deep) flow rates in ml/min/100g
    a, db0, beta : override the fitted values (default: read from `model`)
    Returns g2 of shape (len(flows), 3, len(tau)); tau in seconds.
    """
    a_fit, db0_fit, beta_fit = load_model(model)
    a = a_fit if a is None else a
    db0 = db0_fit if db0 is None else db0
    beta = beta_fit if beta is None else np.broadcast_to(beta, (3,))
    det = data_io.load_detectors(pathlengths, cfg.PHOTONS_PER_SDS)

    out = np.zeros((len(flows), 3, len(tau)))
    for k, (fs, fd) in enumerate(flows):
        aDB_s = cfg.flow_to_alpha_db(fs, a, db0)
        aDB_d = cfg.flow_to_alpha_db(fd, a, db0)
        for s in range(3):
            out[k, s] = calculate_g2(*det[s], aDB_s, aDB_d, tau, beta[s])
    return out


def save_csv(path, flows, g2, tau=data_io.TAU_ARRAY):
    """One row per (flow, SDS): FR_shallow, FR_deep, SDS, tau_0us ... tau_500us."""
    rows = []
    for k, (fs, fd) in enumerate(flows):
        for s, lbl in enumerate(data_io.SDS_LABELS):
            rows.append([fs, fd, lbl, *g2[k, s]])
    cols = ["FR_shallow", "FR_deep", "SDS"] + [f"tau_{t * 1e6:g}us" for t in tau]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
