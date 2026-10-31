"""
MCX module: builds the two-layer flow-phantom voxel grid (paper geometry, see
config/parameters.py) and runs the Monte Carlo photon simulation via pmcxcl (OpenCL --
the Apple Silicon build; `pmcx` itself is CUDA-only). Run ONCE; its output (per-photon
partial pathlength + momentum transfer per layer, per SDS) feeds every flow condition
afterwards through dcs_physics.py.

The installed pmcxcl must be the arm64 build compiled from source (PyPI wheels are
x86_64-only despite the universal2 tag) -- see scripts/setup_env.sh.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg

import pmcxcl


def create_voxel_grid(unitinmm=cfg.VOXEL_SIZE_MM):
    """Voxel labels: 1 shallow layer fluid, 2 deep layer fluid."""
    nx = int(cfg.PHANTOM_X_MM / unitinmm)
    ny = int(cfg.PHANTOM_Y_MM / unitinmm)
    nz = int(round(cfg.FLUID_Z_MM / unitinmm))
    vol = np.full((nx, ny, nz), 2, dtype=np.uint8)
    vol[:, :, :int(cfg.SHALLOW_DEPTH_MM / unitinmm)] = 1
    return vol


def run_simulation(nphoton=cfg.NPHOTON, unitinmm=cfg.VOXEL_SIZE_MM):
    vol = create_voxel_grid(unitinmm)
    fluid = [cfg.MU_A_MM, cfg.MU_S_MM, cfg.ANISOTROPY_G, cfg.REFRACTIVE_INDEX_N]
    prop = [[0.0, 0.0, 1.0, 1.0], fluid, fluid]       # 0 outside, 1 shallow, 2 deep

    nx, ny, nz = vol.shape
    srcpos = [cfg.SOURCE_X_MM / unitinmm, ny / 2.0, 0.0]
    det_radius_vox = cfg.DET_RADIUS_MM / unitinmm
    detpos = [[srcpos[0] + round(d / unitinmm), srcpos[1], 0.0, det_radius_vox] for d in cfg.SDS_MM]

    cfg_dict = {
        "nphoton": int(nphoton),
        "vol": vol,
        "prop": prop,
        "srcpos": srcpos,
        "srcdir": [0.0, 0.0, 1.0],
        "srctype": "pencil",
        "detpos": detpos,
        "issavedet": 1,
        "savedetflag": "dpm",     # detector id, partial pathlength, momentum transfer
        "ismomentum": 1,
        "maxdetphoton": 5_000_000,
        "unitinmm": unitinmm,
        "tstart": 0.0,
        "tend": 5e-9,
        "tstep": 5e-9,
        "bc": cfg.MCX_BC,
    }
    print(f"MCX: grid {vol.shape} (voxel {unitinmm} mm), source {srcpos} vox, "
          f"detectors at SDS {cfg.SDS_MM} mm, {nphoton:.1e} photons, bc {cfg.MCX_BC}")
    res = pmcxcl.run(cfg_dict)

    # pmcxcl returns detp raw as (reclen, n_photons); with savedetflag='dpm':
    # reclen = 1 (detid) + n_media (ppath, voxel units) + n_media (momentum transfer).
    detp = res["detp"]
    n_media = (detp.shape[0] - 1) // 2
    if detp.shape[0] != 1 + 2 * n_media:
        raise RuntimeError(f"Unexpected detp record length {detp.shape[0]}; "
                           "expected 1 + 2*n_media (detid + ppath + mom).")
    detid = detp[0, :]
    ppath = detp[1:1 + n_media, :].T      # (n_photons, n_media)
    mom = detp[1 + n_media:1 + 2 * n_media, :].T

    SHALLOW, DEEP = 0, 1                  # column = medium label - 1
    processed = {}
    for det_idx in range(1, len(detpos) + 1):
        idx = detid == det_idx
        processed[f"det_{det_idx}_Ls"] = (ppath[idx, SHALLOW] * unitinmm).astype(np.float32)   # voxel -> mm
        processed[f"det_{det_idx}_Ld"] = (ppath[idx, DEEP] * unitinmm).astype(np.float32)
        processed[f"det_{det_idx}_Ys"] = mom[idx, SHALLOW].astype(np.float32)
        processed[f"det_{det_idx}_Yd"] = mom[idx, DEEP].astype(np.float32)
        print(f"  SDS {det_idx} cm: {int(idx.sum())} photons detected")
    return processed
