"""
Model A pipeline: one CLI for the three stages.

    python src/main.py run-mcx       # once: MCX photons -> data/pathlengths.npz
    python src/main.py calibrate     # Nelder-Mead fit of (a, D_B0) + metrics + figures
    python src/main.py all           # both
    python src/main.py generate --flows 5,20 12,30    # g2 for chosen (shallow, deep) flows
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import parameters as cfg
import data_io
from calibrate import calibrate, evaluate
from dcs_physics import synth_curves
from plot_results import plot_all


def cmd_run_mcx(args):
    import mcx_engine
    os.makedirs(os.path.dirname(args.pathlengths) or ".", exist_ok=True)
    t0 = time.time()
    paths = mcx_engine.run_simulation(nphoton=args.nphoton)
    np.savez_compressed(args.pathlengths, **paths)
    print(f"saved {args.pathlengths} ({time.time() - t0:.0f} s)")


def cmd_calibrate(args):
    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    trials, _ = data_io.load_testset(args.testset)
    assert not np.any(data_io.excluded_condition(meta["fr_set_s"][trials], meta["fr_set_d"][trials]))
    det = data_io.load_detectors(args.pathlengths, cfg.PHOTONS_PER_SDS)
    print(f"beta = {beta[0]:.3f} / {beta[1]:.3f} / {beta[2]:.3f};  {len(trials)} test-set trials")

    a, db0, res = calibrate(det, meta, beta, tau, g2, trials)
    print(f"a = {a:.4e} mm^2/s per (ml/min/100g), D_B0 = {db0:.4e} mm^2/s "
          f"({res.nfev} evaluations, RMSE {np.sqrt(res.fun):.4f})")

    sim = synth_curves(det, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)
    print("fitting BFI on sim and physical curves...")
    metrics = evaluate(sim, g2[:, :, trials], beta, tau, meta["bfi"][:, trials])
    fmt = lambda v, d=3: " / ".join(f"{x:.{d}f}" for x in v)
    print(f"R2 median   {fmt(metrics['r2_median'])}   (SDS 1 / 2 / 3 cm)")
    print(f"R2 mean     {fmt(metrics['r2_mean'])}")
    print(f"RMSE pooled {fmt(metrics['rmse_pooled'], 4)}")
    print(f"BFI-X ratio median {fmt(metrics['bfiX_ratio_median'], 2)}   mean {fmt(metrics['bfiX_ratio_mean'], 2)}")
    print(f"BFI-Y ratio median {fmt(metrics['bfiY_ratio_median'], 2)}   mean {fmt(metrics['bfiY_ratio_mean'], 2)}")

    os.makedirs(args.out, exist_ok=True)
    met = plot_all(args.out, tau, meta, g2, sim, trials)
    with open(os.path.join(args.out, "model_a.json"), "w") as f:
        json.dump({"a_mm2_s_per_flow": a, "db0_mm2_s": db0, "nelder_mead_evals": int(res.nfev),
                   "pathlengths": os.path.relpath(args.pathlengths, data_io.ROOT),
                   "testset": os.path.relpath(args.testset, data_io.ROOT),
                   "n_trials": int(len(trials)), "n_conditions": int(len(met)),
                   "photons_per_sds": cfg.PHOTONS_PER_SDS, "beta": beta.tolist(),
                   "metrics_per_trial": metrics}, f, indent=2)
    print(f"saved {args.out}")


def cmd_generate(args):
    import pandas as pd
    from generate import generate_g2, save_csv
    if args.flow_csv:
        df = pd.read_csv(args.flow_csv)
        flows = list(zip(df["FR_shallow"].astype(float), df["FR_deep"].astype(float)))
    else:
        flows = [tuple(float(x) for x in f.split(",")) for f in args.flows]
    g2 = generate_g2(flows, pathlengths=args.pathlengths)
    save_csv(args.out, flows, g2)
    print(f"{len(flows)} flow(s) x 3 SDS -> {args.out}")


def cmd_all(args):
    cmd_run_mcx(args)
    cmd_calibrate(args)


def build_parser():
    p = argparse.ArgumentParser(description="Model A: DCS two-layer phantom simulation")
    sub = p.add_subparsers(dest="command", required=True)
    for name, func, help_ in (("run-mcx", cmd_run_mcx, "run MCX once -> pathlengths.npz"),
                              ("calibrate", cmd_calibrate, "fit (a, D_B0), report metrics, plot"),
                              ("all", cmd_all, "run-mcx then calibrate")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--pathlengths", default=data_io.PATHLENGTHS_PATH)
        if name != "calibrate":
            sp.add_argument("--nphoton", type=float, default=cfg.NPHOTON)
        if name != "run-mcx":
            sp.add_argument("--testset", default=data_io.TESTSET_PATH)
            sp.add_argument("--out", default=os.path.join(data_io.ROOT, "final_result"))
        sp.set_defaults(func=func)

    sp = sub.add_parser("generate", help="g2 for chosen (shallow, deep) flows from existing photons")
    grp = sp.add_mutually_exclusive_group(required=True)
    grp.add_argument("--flows", nargs="+", metavar="S,D", help="e.g. 5,20 12,30 (ml/min/100g)")
    grp.add_argument("--flow-csv", help="CSV with columns FR_shallow, FR_deep")
    sp.add_argument("--pathlengths", default=data_io.PATHLENGTHS_PATH)
    sp.add_argument("--out", default=os.path.join(data_io.ROOT, "generated", "g2_generated.csv"))
    sp.set_defaults(func=cmd_generate)
    return p


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
