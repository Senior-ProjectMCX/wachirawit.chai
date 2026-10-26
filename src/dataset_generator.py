"""
Dataset module: loops over every flow condition in the (filtered) flow table
and synthesizes g2(tau) for all 3 SDS, writing CSVs in the same layout as
the physical data so compare.py / bfi_fit.py can read either one.

Ported from core/code/generate_dataset.py, with two fixes:
  - bug 3.8: drives the simulation with FR_meas (measured outlet flow), not
    FR_set (pump preset) -- section 2.3 of the analysis doc showed FR_meas is
    what the paper's own c1/c2/c3 regression was fit against.
  - Goal.md: the (0,0) rows are already filtered out by data_io.load_flow_table()
    before they ever reach this module (deep == 0 with shallow > 0 is kept).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg
import data_io
from dcs_physics import calculate_g2


def generate_synthetic_dataset(pathlengths_path, output_path,
                                a=cfg.DEFAULT_A_MM2_S_PER_FR, db0=cfg.DEFAULT_DB0_MM2_S,
                                max_photons=10_000, skip_deep_flow_zero=cfg.SKIP_DEEP_FLOW_ZERO):
    print("Loading physical g2 (for beta) + flow condition table...")
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    print(f"  beta = {beta[0]:.3f} / {beta[1]:.3f} / {beta[2]:.3f}")

    flow = data_io.load_flow_table(skip_deep_flow_zero=skip_deep_flow_zero)
    n_trials = flow["n_trials"]
    print(f"  a={a:.3e}, D_B0={db0:.3e} mm^2/s   n_trials={n_trials}")

    print(f"Loading MCX pathlengths + momentum from {pathlengths_path}...")
    pl_data = np.load(pathlengths_path)
    for req in ("det_1_Ys", "det_1_Yd"):
        if req not in pl_data:
            raise KeyError(f"'{req}' missing from {pathlengths_path}; "
                            "re-run mcx_engine.py with momentum transfer enabled.")

    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl_data, s + 1, max_photons, rng) for s in range(3)}
    for s in range(3):
        print(f"  SDS {data_io.SDS_LABELS[s]}: {len(det[s][0])} photons after downsample")

    tau = data_io.TAU_ARRAY
    sim_g2 = np.zeros((3, data_io.N_TAU, n_trials), dtype=np.float32)
    fr_s, fr_d = flow["fr_meas_s"], flow["fr_meas_d"]

    print(f"Generating synthetic g2 for {n_trials} trials...")
    for i in range(n_trials):
        aDB_s = cfg.flow_to_alpha_db(fr_s[i], a=a, db0=db0)
        aDB_d = cfg.flow_to_alpha_db(fr_d[i], a=a, db0=db0)
        for s in range(3):
            Ls, Ld, Ys, Yd = det[s]
            sim_g2[s, :, i] = calculate_g2(Ls, Ld, Ys, Yd, aDB_s, aDB_d, tau, beta[s])
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{n_trials}")

    written = data_io.save_synthetic(output_path, sim_g2, flow["trial_index"])
    print("Synthetic dataset written:")
    for f in written:
        print(f"  {f}")
    return sim_g2, flow


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DCS g2 from MCX pathlengths")
    parser.add_argument("--pathlengths", type=str, default="data/pathlengths.npz")
    parser.add_argument("--output", type=str, default="data/Dataset_simulated.csv",
                         help="Stem for synthetic CSVs (writes <stem>_SDS{1,2,3}cm.csv)")
    parser.add_argument("--a", type=float, default=cfg.DEFAULT_A_MM2_S_PER_FR)
    parser.add_argument("--db0", type=float, default=cfg.DEFAULT_DB0_MM2_S)
    parser.add_argument("--max-photons", type=int, default=10_000)
    parser.add_argument("--include-deep-zero", action="store_true",
                         help="Do NOT filter out the (0,0) trials (overrides Goal.md default)")
    args = parser.parse_args()

    generate_synthetic_dataset(args.pathlengths, args.output, a=args.a, db0=args.db0,
                                max_photons=args.max_photons,
                                skip_deep_flow_zero=not args.include_deep_zero)


if __name__ == "__main__":
    main()
