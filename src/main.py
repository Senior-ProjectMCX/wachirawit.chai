"""
Main module: single CLI entry point that chains the pipeline stages.

    python src/main.py run-mcx            --nphoton 1e8
    python src/main.py generate-dataset
    python src/main.py validate-geometry  --synthetic data/Dataset_simulated.csv
    python src/main.py compare
    python src/main.py all                # run-mcx -> generate-dataset -> validate-geometry -> compare

See the data-flow diagram in the approved plan / "DCS Two-Layer Flow Phantom
Simulation Analysis" note in Obsidian for how these stages relate.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import parameters as cfg
import data_io
import mcx_engine
import dataset_generator
import bfi_fit
import compare as compare_mod


def cmd_run_mcx(args):
    out_dir = os.path.dirname(args.pathlengths)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    import numpy as np
    paths = mcx_engine.run_simulation(nphoton=args.nphoton, geometry=args.geometry,
                                       dry_run=args.dry_run)
    if paths is not None:
        np.savez_compressed(args.pathlengths, **paths)
        print(f"Pathlength data saved to {args.pathlengths}")


def cmd_generate_dataset(args):
    dataset_generator.generate_synthetic_dataset(
        args.pathlengths, args.output, a=args.a, db0=args.db0,
        max_photons=args.max_photons, skip_deep_flow_zero=not args.include_deep_zero)


def cmd_validate_geometry(args):
    import numpy as np
    meta = data_io.load_physical_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    tau = data_io.TAU_ARRAY
    print(f"Coherence factors beta = {beta[0]:.3f} / {beta[1]:.3f} / {beta[2]:.3f} "
          f"(expect ~0.407/0.494/0.490)")

    keep = meta["fr_set_d"] != 0.0 if not args.include_deep_zero else np.ones(meta["n_trials"], bool)
    fr_s, fr_d = meta["fr_meas_s"][keep], meta["fr_meas_d"][keep]

    gt = bfi_fit.regress_c2c1(meta["bfi"][:, keep], fr_s, fr_d)
    bfi_fit.print_regression("[1] PAPER BFI (metadata) vs FR_meas:", gt)

    if not args.synthetic:
        return

    # Our semi-infinite fitter reproduces the paper's BFI trends (R~0.99) but
    # not its absolute per-SDS scale, so the fair gate compares synthetic and
    # physical curves put through the SAME fitter (as core/code/c2c1_test.py did).
    print("\n[2] Fitting BFI from PHYSICAL g2 with our fitter (same-fitter reference)...")
    bfi_phys = bfi_fit.fit_bfi_all(phys_g2[:, :, keep], beta, tau)
    res_phys = bfi_fit.regress_c2c1(bfi_phys, fr_s, fr_d)
    bfi_fit.print_regression("    c2/c1 from physical g2, our fitter:", res_phys)
    phys_ratio = {s: res_phys[s][4] for s in range(3)}

    print(f"\n[3] Fitting BFI from SYNTHETIC g2 ({args.synthetic})...")
    sim_g2 = data_io.load_synthetic_g2(args.synthetic)
    flow = data_io.load_flow_table(skip_deep_flow_zero=not args.include_deep_zero)
    bfi_sim = bfi_fit.fit_bfi_all(sim_g2, beta, tau)
    res_sim = bfi_fit.regress_c2c1(bfi_sim, flow["fr_meas_s"], flow["fr_meas_d"])
    bfi_fit.print_regression("    c2/c1 from SYNTHETIC g2 -- THE GEOMETRY GATE:", res_sim,
                              reference=phys_ratio, ref_label="phys-fit")
    passed = bfi_fit.geometry_gate_passed(res_sim, target=phys_ratio)
    print(f"\n  Geometry gate vs same-fitter physical reference: {'PASS' if passed else 'FAIL'}")
    print(f"  (paper c2/c1 targets for context: {cfg.TARGET_C2_C1})")


def cmd_compare(args):
    compare_mod.run_comparison(args.pathlengths, args.output_plot,
                                metrics_output_path=args.metrics_plot,
                                test_frac=args.test_frac, seed=args.seed,
                                skip_deep_flow_zero=not args.include_deep_zero)


def cmd_all(args):
    cmd_run_mcx(args)
    if args.dry_run:
        return
    cmd_generate_dataset(args)
    args.synthetic = args.output
    cmd_validate_geometry(args)
    cmd_compare(args)


def build_parser():
    p = argparse.ArgumentParser(description="wachirawit.chai DCS two-layer phantom pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    p_mcx = sub.add_parser("run-mcx", help="Run MCX once to produce pathlengths.npz")
    p_mcx.add_argument("--nphoton", type=float, default=1e8)
    p_mcx.add_argument("--geometry", type=str, default=cfg.DEFAULT_GEOMETRY,
                        choices=["simplified", "realistic"])
    p_mcx.add_argument("--pathlengths", type=str, default="data/pathlengths.npz")
    p_mcx.add_argument("--dry-run", action="store_true")
    p_mcx.set_defaults(func=cmd_run_mcx)

    p_gen = sub.add_parser("generate-dataset", help="Synthesize g2 for every flow condition")
    p_gen.add_argument("--pathlengths", type=str, default="data/pathlengths.npz")
    p_gen.add_argument("--output", type=str, default="data/Dataset_simulated.csv")
    p_gen.add_argument("--a", type=float, default=cfg.DEFAULT_A_MM2_S_PER_FR)
    p_gen.add_argument("--db0", type=float, default=cfg.DEFAULT_DB0_MM2_S)
    p_gen.add_argument("--max-photons", type=int, default=10_000)
    p_gen.add_argument("--include-deep-zero", action="store_true")
    p_gen.set_defaults(func=cmd_generate_dataset)

    p_val = sub.add_parser("validate-geometry", help="c2/c1 geometry gate (run before calibrating)")
    p_val.add_argument("--synthetic", type=str, default=None)
    p_val.add_argument("--include-deep-zero", action="store_true")
    p_val.set_defaults(func=cmd_validate_geometry)

    p_cmp = sub.add_parser("compare", help="Calibrate (a, D_B0) + report R^2/MAE/RMSE vs physical data")
    p_cmp.add_argument("--pathlengths", type=str, default="data/pathlengths.npz")
    p_cmp.add_argument("--output-plot", type=str, default="plots/g2_comparison.png")
    p_cmp.add_argument("--metrics-plot", type=str, default="plots/g2_metrics.png")
    p_cmp.add_argument("--test-frac", type=float, default=0.30)
    p_cmp.add_argument("--seed", type=int, default=0)
    p_cmp.add_argument("--include-deep-zero", action="store_true")
    p_cmp.set_defaults(func=cmd_compare)

    p_all = sub.add_parser("all", help="run-mcx -> generate-dataset -> validate-geometry -> compare")
    p_all.add_argument("--nphoton", type=float, default=1e8)
    p_all.add_argument("--geometry", type=str, default=cfg.DEFAULT_GEOMETRY,
                        choices=["simplified", "realistic"])
    p_all.add_argument("--pathlengths", type=str, default="data/pathlengths.npz")
    p_all.add_argument("--output", type=str, default="data/Dataset_simulated.csv")
    p_all.add_argument("--a", type=float, default=cfg.DEFAULT_A_MM2_S_PER_FR)
    p_all.add_argument("--db0", type=float, default=cfg.DEFAULT_DB0_MM2_S)
    p_all.add_argument("--max-photons", type=int, default=10_000)
    p_all.add_argument("--output-plot", type=str, default="plots/g2_comparison.png")
    p_all.add_argument("--metrics-plot", type=str, default="plots/g2_metrics.png")
    p_all.add_argument("--test-frac", type=float, default=0.30)
    p_all.add_argument("--seed", type=int, default=0)
    p_all.add_argument("--dry-run", action="store_true")
    p_all.add_argument("--include-deep-zero", action="store_true")
    p_all.set_defaults(func=cmd_all)

    return p


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
