"""
Data module: all file I/O for the pipeline in one place.

  - core/data/{metadata.csv, g2_SDS{1,2,3}cm.csv}: the PHYSICAL lab measurements
    (ground truth). Trial_Index in the test set is the row index into these files.
  - dataset/testset_closest_k5.csv: the 5 trials per flow condition whose measured
    flow is closest to the preset (scripts/make_testset.py), 132 conditions, (0,0) excluded.
  - data/pathlengths.npz: MCX photon pathlengths + momentum transfer (mcx_engine.py).
"""
import os

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SRC_DIR)                             # .../wachirawit.chai
CORE_DATA_DIR = os.path.join(os.path.dirname(ROOT), "core", "data")
TESTSET_PATH = os.path.join(ROOT, "dataset", "testset_closest_k5.csv")
PATHLENGTHS_PATH = os.path.join(ROOT, "data", "pathlengths.npz")

N_TAU = 501
TAU_ARRAY = np.arange(N_TAU) * 1e-6          # seconds, 0..500us step 1us
SDS_LABELS = ["1cm", "2cm", "3cm"]


def excluded_condition(fr_set_s, fr_set_d):
    """True for the (0, 0) preset condition, the only one skipped per Goal.md."""
    return (np.asarray(fr_set_s) == 0.0) & (np.asarray(fr_set_d) == 0.0)


def load_physical_metadata(data_dir=CORE_DATA_DIR):
    """Load core/data/metadata.csv. Returns dict of 1D arrays + n_trials."""
    df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    return {
        "fr_set_s": df["FR_set_shallow_ml_min_100g"].to_numpy(float),
        "fr_set_d": df["FR_set_deep_ml_min_100g"].to_numpy(float),
        "fr_meas_s": df["FR_meas_shallow_ml_min_100g"].to_numpy(float),
        "fr_meas_d": df["FR_meas_deep_ml_min_100g"].to_numpy(float),
        "bfi": np.vstack([
            df["BFI_SDS1cm_1e-9cm2_s"].to_numpy(float),
            df["BFI_SDS2cm_1e-9cm2_s"].to_numpy(float),
            df["BFI_SDS3cm_1e-9cm2_s"].to_numpy(float),
        ]),
        "n_trials": len(df),
    }


def load_physical_g2(data_dir=CORE_DATA_DIR):
    """Load g2_SDS{1,2,3}cm.csv -> array shape (3, N_TAU, n_trials)."""
    curves = []
    for lbl in SDS_LABELS:
        df = pd.read_csv(os.path.join(data_dir, f"g2_SDS{lbl}.csv"))
        tau_cols = [c for c in df.columns if c.startswith("tau_")]
        assert len(tau_cols) == N_TAU, f"expected {N_TAU} tau columns, got {len(tau_cols)}"
        curves.append(df[tau_cols].to_numpy(float).T)   # (N_TAU, n_trials)
    return np.stack(curves, axis=0)


def extract_beta(physical_g2):
    """Coherence factor per SDS from physical g2 at tau=0: beta = mean(g2(0)) - 1."""
    return np.array([np.mean(physical_g2[s, 0, :]) - 1.0 for s in range(physical_g2.shape[0])])


def load_testset(path=TESTSET_PATH):
    """Test-set trial indices (sorted) and the table itself."""
    ts = pd.read_csv(path).sort_values("Trial_Index").reset_index(drop=True)
    return ts["Trial_Index"].to_numpy(int), ts


def load_detectors(path, max_photons, seed=42):
    """{s: (Ls, Ld, Ys, Yd)} for SDS s = 0, 1, 2, each subsampled to max_photons with a
    SHARED index so the 4 arrays stay per-photon-consistent."""
    pl = np.load(path)
    rng = np.random.default_rng(seed)
    det = {}
    for s in range(3):
        arrs = [pl[f"det_{s + 1}_{k}"] for k in ("Ls", "Ld", "Ys", "Yd")]
        if len(arrs[0]) > max_photons:
            idx = rng.choice(len(arrs[0]), max_photons, replace=False)
            arrs = [x[idx] for x in arrs]
        det[s] = tuple(arrs)
    return det
