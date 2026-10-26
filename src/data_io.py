"""
Data module: all file I/O for the pipeline in one place.

Two distinct data sources are used on purpose (see "Data set definition" in
Obsidian):

  1. wachirawit.chai/dataset/Dataset_metadata.csv
     -> the table of (shallow, deep) flow conditions to DRIVE the simulation.
        The (0, 0) condition (no flow in either layer) is filtered out here by
        default, per Goal.md ("ข้ามกรณี Flow rate ที่ชั้นลึกตื้น = 0"; clarified
        2026-10-15: only (0, 0) is skipped, deep == 0 with shallow > 0 is kept).

  2. core/data/{metadata.csv, g2_SDS{1,2,3}cm.csv}
     -> the PHYSICAL lab measurements used as ground truth for compare.py /
        calibration (chosen over the paper's published c1/c2/c3 equations,
        per the 2026-09-30 planning decision -- those equations are used only
        as the secondary c2/c1 geometry gate, see bfi_fit.py).
"""
import os

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WACHIRAWIT_ROOT = os.path.dirname(_SRC_DIR)                 # .../wachirawit.chai
PROJECT_ROOT = os.path.dirname(WACHIRAWIT_ROOT)              # .../project

WACHIRAWIT_DATASET_DIR = os.path.join(WACHIRAWIT_ROOT, "dataset")
CORE_DATA_DIR = os.path.join(PROJECT_ROOT, "core", "data")
OUTPUT_DATA_DIR = os.path.join(WACHIRAWIT_ROOT, "data")

N_TAU = 501


def excluded_condition(fr_set_s, fr_set_d):
    """True for the (0, 0) preset condition, the only one skipped per Goal.md."""
    return (np.asarray(fr_set_s) == 0.0) & (np.asarray(fr_set_d) == 0.0)

TAU_ARRAY = np.arange(N_TAU) * 1e-6          # seconds, 0..500us step 1us
SDS_LABELS = ["1cm", "2cm", "3cm"]


# ============================================================
# 1. FLOW CONDITION TABLE (drives the simulation)
# ============================================================

def load_flow_table(skip_deep_flow_zero=True, data_dir=WACHIRAWIT_DATASET_DIR):
    """
    Load wachirawit.chai/dataset/Dataset_metadata.csv.

    Returns a dict of 1D arrays (length = n_trials after filtering):
        trial_index, fr_set_s, fr_set_d, fr_meas_s, fr_meas_d,
        bfi_1cm, bfi_2cm, bfi_3cm, n_trials
    """
    df = pd.read_csv(os.path.join(data_dir, "Dataset_metadata.csv"))

    if skip_deep_flow_zero:
        before = len(df)
        df = df[~excluded_condition(df["Shallow_Flow_Preset_Pump"], df["Deep_Flow_Preset_Pump"])].reset_index(drop=True)
        print(f"load_flow_table: skipped {before - len(df)} (0,0) trials "
              f"({len(df)} remain) per Goal.md")

    return {
        "trial_index": df["Trial_Index"].to_numpy(int),
        "fr_set_s": df["Shallow_Flow_Preset_Pump"].to_numpy(float),
        "fr_set_d": df["Deep_Flow_Preset_Pump"].to_numpy(float),
        "fr_meas_s": df["Shallow_Flow_Measured_Outlet"].to_numpy(float),
        "fr_meas_d": df["Deep_Flow_Measured_Outlet"].to_numpy(float),
        "bfi_1cm": df["BFI_1cm_SDS"].to_numpy(float),
        "bfi_2cm": df["BFI_2cm_SDS"].to_numpy(float),
        "bfi_3cm": df["BFI_3cm_SDS"].to_numpy(float),
        "n_trials": len(df),
    }


# ============================================================
# 2. PHYSICAL REFERENCE DATA (ground truth for compare.py)
# ============================================================

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
    """Coherence factor per SDS from physical g2 at tau=0: beta = mean(g2(0)) - 1.
    Expected ~0.407 / 0.494 / 0.490 for this dataset."""
    return np.array([np.mean(physical_g2[s, 0, :]) - 1.0 for s in range(physical_g2.shape[0])])


# ============================================================
# 3. MCX PATHLENGTHS (produced once by mcx_engine.py)
# ============================================================

def load_pathlengths(path=None):
    path = path or os.path.join(OUTPUT_DATA_DIR, "pathlengths.npz")
    return np.load(path)


# ============================================================
# 4. SYNTHETIC g2 OUTPUT (produced by dataset_generator.py)
# ============================================================

def save_synthetic(path, sim_g2, trial_index):
    """Write <stem>_SDS{1,2,3}cm.csv next to `path`. sim_g2 shape (3, N_TAU, n_trials)."""
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    tau_cols = [f"tau_{i}us" for i in range(N_TAU)]
    written = []
    for s, lbl in enumerate(SDS_LABELS):
        df = pd.DataFrame(sim_g2[s].T, columns=tau_cols)
        df.insert(0, "trial", trial_index)
        fpath = os.path.join(out_dir, f"{stem}_SDS{lbl}.csv")
        df.to_csv(fpath, index=False)
        written.append(fpath)
    return written


def downsample_detector(pl_data, det_idx, max_photons, rng):
    """Return aligned (Ls, Ld, Ys, Yd) for one detector, subsampled with a
    SHARED index so the 4 arrays stay per-photon-consistent."""
    Ls = pl_data[f"det_{det_idx}_Ls"]
    Ld = pl_data[f"det_{det_idx}_Ld"]
    Ys = pl_data[f"det_{det_idx}_Ys"]
    Yd = pl_data[f"det_{det_idx}_Yd"]
    if len(Ls) > max_photons:
        idx = rng.choice(len(Ls), max_photons, replace=False)
        Ls, Ld, Ys, Yd = Ls[idx], Ld[idx], Ys[idx], Yd[idx]
    return Ls, Ld, Ys, Yd


def load_synthetic_g2(path, data_dir=None):
    out_dir = data_dir or (os.path.dirname(os.path.abspath(path)) or ".")
    stem = os.path.splitext(os.path.basename(path))[0]
    curves = []
    for lbl in SDS_LABELS:
        df = pd.read_csv(os.path.join(out_dir, f"{stem}_SDS{lbl}.csv"))
        tau_cols = [c for c in df.columns if c.startswith("tau_")]
        curves.append(df[tau_cols].to_numpy(float).T)
    return np.stack(curves, axis=0)
