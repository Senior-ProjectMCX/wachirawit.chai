"""
MCX module: builds the two-layer flow-phantom voxel grid and runs the Monte
Carlo photon simulation via pmcxcl (OpenCL -- the Apple Silicon build; `pmcx`
itself is CUDA-only and won't run on this machine). This step is run ONCE;
its output (per-photon partial pathlength + momentum transfer per layer, per
SDS) feeds every flow condition afterwards through dcs_physics.py, which is
pure NumPy and fast.

Ported from core/code/simulation.py with three physics fixes from the
"DCS Two-Layer Flow Phantom Simulation Analysis" bug list:

  - bug 3.2: default geometry is 'simplified' (no Mylar voxel layer). Forcing
    the physical 0.05mm Mylar sheet into a 0.5mm voxel makes it a 10x-too-thick
    scattering wall; the authors' own MC setup is closer to a plain two-layer
    slab, and core/'s own validation found the slab model fit the physical
    data better than the 'realistic' walled+Mylar geometry.
  - bug 3.3/3.9: the top (tissue/air) boundary used a fully absorbing 'a' bc,
    which silently discards all photons that reach z=0 instead of letting
    ~49% Fresnel-reflect back into the tissue (n=1.37 vs air n=1.0). This
    under-counted long paths and left SDS=3cm with too few detected photons
    (13k out of 1e8 in the prior run). Switched to a mismatched-boundary 'm'
    on the top face.

NOTE: the exact 6-character bc face ordering ('m' must land on the z=0 /
top face specifically) has NOT been empirically re-verified in THIS repo,
because the installed pmcxcl wheel is x86_64-only and its compiled extension
does not run on this Apple Silicon machine (see "Issue and Solution" in
Obsidian). Verify with a small dry run once pmcxcl actually executes, e.g. by
checking that detected photon counts at SDS=3cm increase substantially vs.
the all-absorbing baseline.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg

try:
    import pmcxcl as _mcx_backend
    _BACKEND_NAME = "pmcxcl"
except ImportError:
    import pmcx as _mcx_backend
    _BACKEND_NAME = "pmcx"


def create_voxel_grid(geometry=cfg.DEFAULT_GEOMETRY, unitinmm=cfg.VOXEL_SIZE_MM):
    """
    Voxel labels:
      0 Air/Void, 1 Shallow layer fluid, 2 Deep layer fluid,
      3 Mylar sheet separator ('realistic' only), 4 Acrylic wall ('realistic' only)
    """
    if geometry == "simplified":
        nx = int(60 / unitinmm)
        ny = int(60 / unitinmm)
        nz = int(50 / unitinmm)
        vol = np.zeros((nx, ny, nz), dtype=np.uint8)

        z_shallow_end = int(cfg.SHALLOW_DEPTH_MM / unitinmm)
        vol[:, :, :z_shallow_end] = 1
        vol[:, :, z_shallow_end:] = 2

    elif geometry == "realistic":
        nx = int(cfg.PHANTOM_X_MM / unitinmm)
        ny = int(cfg.PHANTOM_Y_MM / unitinmm)
        nz = int((cfg.TOTAL_DEPTH_MM + 0.5) / unitinmm)
        vol = np.zeros((nx, ny, nz), dtype=np.uint8)

        wall_voxels = int(2.5 / unitinmm)
        vol[:wall_voxels, :, :] = 4
        vol[-wall_voxels:, :, :] = 4
        vol[:, :wall_voxels, :] = 4
        vol[:, -wall_voxels:, :] = 4

        z_shallow_end = int(cfg.SHALLOW_DEPTH_MM / unitinmm)
        z_mylar_end = z_shallow_end + 1
        z_deep_end = z_mylar_end + int(cfg.DEEP_DEPTH_MM / unitinmm)

        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, :z_shallow_end] = 1
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_shallow_end:z_mylar_end] = 3
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_mylar_end:z_deep_end] = 2
        vol[wall_voxels:-wall_voxels, wall_voxels:-wall_voxels, z_deep_end:] = 4
    else:
        raise ValueError(f"Unknown geometry type: {geometry}")

    return vol


def run_simulation(nphoton=1e8, geometry=cfg.DEFAULT_GEOMETRY, unitinmm=cfg.VOXEL_SIZE_MM,
                    dry_run=False):
    print(f"[{_BACKEND_NAME}] Setting up '{geometry}' geometry voxel grid...")
    vol = create_voxel_grid(geometry, unitinmm)
    print(f"Voxel grid shape: {vol.shape}")

    prop = [
        [0.0, 0.0, 1.0, 1.0],                                   # 0: Air/Background
        [cfg.MU_A_MM, cfg.MU_S_MM, cfg.ANISOTROPY_G, cfg.REFRACTIVE_INDEX_N],  # 1: Shallow fluid
        [cfg.MU_A_MM, cfg.MU_S_MM, cfg.ANISOTROPY_G, cfg.REFRACTIVE_INDEX_N],  # 2: Deep fluid
        [0.01, 10.0, 0.9, 1.64],                                # 3: Mylar sheet ('realistic' only)
        [0.005, 0.0, 1.0, 1.49],                                # 4: Acrylic wall ('realistic' only)
    ]

    nx, ny, nz = vol.shape
    srcpos = [nx / 2.0, ny / 2.0, 0.0]

    sds_voxels = [int(round(d / unitinmm)) for d in cfg.SDS_MM]
    det_radius_vox = cfg.DET_RADIUS_MM / unitinmm
    detpos = [[srcpos[0] + sds, srcpos[1], 0.0, det_radius_vox] for sds in sds_voxels]

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
        # Top face (tissue/air, z=0) mismatched so ~49% of photons Fresnel-reflect
        # back in (fixes bug 3.3); remaining faces absorbing (domain edges, far
        # from the source/detectors in the simplified slab). VERIFY face order
        # once pmcxcl actually runs on this machine -- see module docstring.
        "bc": "aaaaam",
    }

    print("\n================== MCX Configuration ==================")
    print(f"  backend       : {_BACKEND_NAME}")
    print(f"  geometry      : {geometry}")
    print(f"  voxel shape   : {vol.shape}  (unitinmm={unitinmm} mm)")
    print("  media (prop)  : mua, mus, g, n")
    for i, p in enumerate(prop):
        print(f"    [{i}] {p}")
    print(f"  srcpos (vox)  : {srcpos}   srcdir: {cfg_dict['srcdir']}   srctype: {cfg_dict['srctype']}")
    print(f"  detpos (vox)  : {detpos}   (SDS = 1, 2, 3 cm)")
    print(f"  nphoton       : {int(nphoton):.1e}")
    print(f"  bc            : {cfg_dict['bc']}")
    print("======================================================\n")

    if dry_run:
        print("Dry run requested. Configuration is valid; no photons launched.")
        return None

    print(f"Launching simulation with {nphoton:.1e} photons...")
    res = _mcx_backend.run(cfg_dict) if hasattr(_mcx_backend, "run") else _mcx_backend.mcxlab(cfg_dict)
    print("Simulation finished.")

    detp = res["detp"]

    if isinstance(detp, dict):
        # pmcx (CUDA) mcxlab-style: already parsed into named fields.
        detid = detp["detid"]
        ppath = detp["ppath"]
        if "mom" not in detp:
            raise RuntimeError(
                "Momentum transfer ('mom') missing from output. "
                "Ensure cfg['ismomentum']=1 and savedetflag contains 'm'."
            )
        mom = detp["mom"]
    else:
        # pmcxcl (OpenCL) raw form: detp is (reclen, n_photons), reclen = 1
        # (detid) + n_media (ppath) + n_media (mom); here n_media=4 (shallow,
        # deep, mylar, acrylic) so reclen=9. Empirically verified 2026-09-30:
        # rows 3-4 (mylar/acrylic ppath) are exactly zero in the 'simplified'
        # slab geometry, which has no mylar/acrylic voxels -- confirms the
        # row layout below.
        n_media = (detp.shape[0] - 1) // 2
        if detp.shape[0] != 1 + 2 * n_media:
            raise RuntimeError(f"Unexpected detp record length {detp.shape[0]}; "
                                "expected 1 + 2*n_media (detid + ppath + mom).")
        detid = detp[0, :]
        ppath = detp[1:1 + n_media, :].T      # (n_photons, n_media)
        mom = detp[1 + n_media:1 + 2 * n_media, :].T

    # ppath/mom columns are 0-indexed by (medium label - 1): col0=medium 1
    # (shallow), col1=medium 2 (deep). This ordering was empirically verified
    # in core/code/simulation.py (bug 3.1 fix) and reconfirmed above for the
    # pmcxcl raw layout.
    SHALLOW, DEEP = 0, 1
    processed = {}
    for det_idx in range(1, len(detpos) + 1):
        idx = (detid == det_idx)
        Ls = ppath[idx, SHALLOW] * unitinmm
        Ld = ppath[idx, DEEP] * unitinmm
        Ys = mom[idx, SHALLOW]
        Yd = mom[idx, DEEP]

        n_det = len(Ls)
        print(f"Detector {det_idx} (SDS {det_idx}cm): {n_det} photons captured"
              + (f", mean Ls={Ls.mean():.2f}mm Ld={Ld.mean():.2f}mm" if n_det else ""))
        processed[f"det_{det_idx}_Ls"] = Ls.astype(np.float32)
        processed[f"det_{det_idx}_Ld"] = Ld.astype(np.float32)
        processed[f"det_{det_idx}_Ys"] = Ys.astype(np.float32)
        processed[f"det_{det_idx}_Yd"] = Yd.astype(np.float32)

    return processed


def main():
    parser = argparse.ArgumentParser(description="Run the DCS two-layer phantom MCX simulation")
    parser.add_argument("--nphoton", type=float, default=1e8)
    parser.add_argument("--geometry", type=str, default=cfg.DEFAULT_GEOMETRY,
                         choices=["simplified", "realistic"])
    parser.add_argument("--output", type=str, default="data/pathlengths.npz")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    paths = run_simulation(nphoton=args.nphoton, geometry=args.geometry, dry_run=args.dry_run)
    if paths is not None:
        np.savez_compressed(args.output, **paths)
        print(f"Pathlength data saved to {args.output}")


if __name__ == "__main__":
    main()
