"""
Compare module: calibrates the single flow->D_B scale factor `a` (and the
zero-flow baseline D_B0) against the PHYSICAL lab g2 curves in core/data/
(the ground truth chosen on 2026-09-30, over the paper's published c1/c2/c3
equations -- those are used only as the secondary geometry gate in
bfi_fit.py), then reports held-out R^2 / MAE / RMSE per SDS.

Ported from core/code/compare.py, with:
  - bug 3.6 preserved as a fix: ONE shared scale `a` for both layers, not two
    independent scale_s/scale_d (that was overfitting, see analysis doc 3.6).
  - Deep_Flow==0 trials excluded from the train/test combos by default, to
    match the 2026-09-30 revision of Goal.md.
  - R^2 (not just MAE/RMSE) is now the headline metric, per the 2026-09-30
    clarification that Goal.md's "RMSE ~= 0.95" actually meant R^2 ~= 0.95.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg
import data_io
from dcs_physics import calculate_g2


def _combo_split(fr_s, fr_d, test_frac=0.30, seed=0):
    """Split trial indices by unique (FR_s, FR_d) combo so no combo leaks
    across train/test. Returns (train_idx, test_idx, n_combos)."""
    combos = {}
    for i, key in enumerate(zip(np.round(fr_s, 6), np.round(fr_d, 6))):
        combos.setdefault(key, []).append(i)
    keys = list(combos.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac)))
    test_keys = set(keys[:n_test])
    train_idx, test_idx = [], []
    for key, idxs in combos.items():
        (test_idx if key in test_keys else train_idx).extend(idxs)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx)), len(keys)


def _synth_curves(det, fr_s, fr_d, beta, tau, trials, a, db0):
    out = np.zeros((3, len(tau), len(trials)))
    for j, i in enumerate(trials):
        aDB_s = cfg.flow_to_alpha_db(fr_s[i], a=a, db0=db0)
        aDB_d = cfg.flow_to_alpha_db(fr_d[i], a=a, db0=db0)
        for s in range(3):
            Ls, Ld, Ys, Yd = det[s]
            out[s, :, j] = calculate_g2(Ls, Ld, Ys, Yd, aDB_s, aDB_d, tau, beta[s])
    return out


def _r2(sim, ref):
    """R^2 pooling all (tau, trial) points for one SDS: 1 - SS_res/SS_tot."""
    resid = ref - sim
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def run_comparison(pathlengths_path, plot_output_path, metrics_output_path="plots/g2_metrics.png",
                    test_frac=0.30, seed=0, cal_max_photons=5000,
                    skip_deep_flow_zero=cfg.SKIP_DEEP_FLOW_ZERO):
    meta = data_io.load_physical_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    tau = data_io.TAU_ARRAY
    fr_s, fr_d = meta["fr_meas_s"], meta["fr_meas_d"]

    keep = np.ones(meta["n_trials"], dtype=bool)
    if skip_deep_flow_zero:
        keep = meta["fr_set_d"] != 0.0
        print(f"Excluding {np.sum(~keep)} physical trials with Deep_Flow==0 "
              f"({np.sum(keep)} remain), per Goal.md")

    pl = np.load(pathlengths_path)
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, cal_max_photons, rng) for s in range(3)}

    kept_idx = np.where(keep)[0]
    train_sub, test_sub, n_combos = _combo_split(fr_s[kept_idx], fr_d[kept_idx], test_frac, seed)
    train_idx, test_idx = kept_idx[train_sub], kept_idx[test_sub]
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test overlap!"
    print(f"{n_combos} unique flow combos -> {len(train_idx)} train / {len(test_idx)} test trials")

    # --- Calibrate (a, D_B0) jointly on TRAIN combos, one trial per combo ---
    seen, cal_trials = set(), []
    for i in train_idx:
        key = (round(fr_s[i], 6), round(fr_d[i], 6))
        if key not in seen:
            seen.add(key)
            cal_trials.append(i)
    cal_trials = np.array(cal_trials)

    def cal_loss(log_params):
        a, db0 = 10.0 ** log_params[0], 10.0 ** log_params[1]
        sim = _synth_curves(det, fr_s, fr_d, beta, tau, cal_trials, a, db0)
        ref = phys_g2[:, :, cal_trials]
        return np.mean((sim - ref) ** 2)

    print(f"Calibrating (a, D_B0) on {len(cal_trials)} train combos...")
    x0 = [np.log10(cfg.DEFAULT_A_MM2_S_PER_FR), np.log10(cfg.DEFAULT_DB0_MM2_S)]
    res = minimize(cal_loss, x0, method="Nelder-Mead",
                    bounds=[(-9.0, -3.0), (-9.0, -5.0)])
    a_star, db0_star = 10.0 ** res.x[0], 10.0 ** res.x[1]
    print(f"  calibrated a={a_star:.4e} mm^2/s per (ml/min/100g)   D_B0={db0_star:.4e} mm^2/s")

    # --- Report R^2 / MAE / RMSE on TEST trials only ---
    sim_test = _synth_curves(det, fr_s, fr_d, beta, tau, test_idx, a_star, db0_star)
    ref_test = phys_g2[:, :, test_idx]
    r2 = np.zeros(3)
    mae = np.zeros(3)
    rmse = np.zeros(3)
    per_trial_mae = np.zeros((3, len(test_idx)))
    print("\n============ Held-out (TEST) curve metrics ============")
    for s in range(3):
        diff = sim_test[s] - ref_test[s]
        r2[s] = _r2(sim_test[s], ref_test[s])
        mae[s] = np.mean(np.abs(diff))
        rmse[s] = np.sqrt(np.mean(diff ** 2))
        per_trial_mae[s] = np.mean(np.abs(diff), axis=0)
        flag = "PASS" if r2[s] >= 0.95 else "below target"
        print(f"  SDS {data_io.SDS_LABELS[s]} - R^2: {r2[s]:.4f} ({flag}) | "
              f"MAE: {mae[s]:.5f} | RMSE: {rmse[s]:.5f}")
    print("=======================================================")

    _overlay_plot(det, fr_s, fr_d, beta, tau, phys_g2, test_idx, a_star, db0_star, plot_output_path)
    _metrics_plot(r2, mae, rmse, per_trial_mae, fr_s[test_idx] + fr_d[test_idx], metrics_output_path)
    return {"a": a_star, "db0": db0_star, "r2": r2, "mae": mae, "rmse": rmse}


def _overlay_plot(det, fr_s, fr_d, beta, tau, phys_g2, test_idx, a_star, db0_star, plot_output_path):
    order = test_idx[np.argsort(fr_s[test_idx] + fr_d[test_idx])]
    picks = order[np.linspace(0, len(order) - 1, min(4, len(order))).astype(int)]
    tau_us = np.arange(data_io.N_TAU)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax_i, i in enumerate(picks):
        ax = axes[ax_i]
        sim = _synth_curves(det, fr_s, fr_d, beta, tau, [i], a_star, db0_star)[:, :, 0]
        for s, lbl in enumerate(data_io.SDS_LABELS):
            ax.plot(tau_us, phys_g2[s, :, i], color=colors[s], ls="-", alpha=0.8, label=f"Phys {lbl}")
            ax.plot(tau_us, sim[s], color=colors[s], ls="--", lw=2, label=f"Sim {lbl}")
        ax.set_title(f"[TEST] Shallow={fr_s[i]:.1f}, Deep={fr_d[i]:.1f} ml/min/100g")
        ax.set_xlabel(r"Delay $\tau$ ($\mu$s)")
        ax.set_ylabel(r"$g_2(\tau)$")
        ax.grid(True, ls=":", alpha=0.6)
        if ax_i == 0:
            ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_dir = os.path.dirname(plot_output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(plot_output_path, dpi=200)
    print(f"Comparison plot saved to {plot_output_path}")


def _metrics_plot(r2, mae, rmse, per_trial_mae, total_flow, metrics_output_path):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    order = np.argsort(total_flow)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(3)
    bars = axA.bar(x, r2, 0.5, color=["#4c72b0" if v >= 0.95 else "#c44e52" for v in r2])
    axA.axhline(0.95, color="k", ls="--", lw=1, label="target R^2=0.95")
    axA.set_xticks(x)
    axA.set_xticklabels([f"{l} SDS" for l in data_io.SDS_LABELS])
    axA.set_ylim(0, 1.05)
    axA.set_ylabel("R^2 (held-out test)")
    axA.set_title("(A) Held-out R^2 per SDS")
    axA.legend()
    axA.grid(True, axis="y", ls=":", alpha=0.6)
    for r in bars:
        axA.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                      textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

    for s, lbl in enumerate(data_io.SDS_LABELS):
        axB.plot(np.arange(len(order)), per_trial_mae[s, order],
                  color=colors[s], lw=1.2, alpha=0.85, label=f"{lbl} SDS")
    axB.set_xlabel("test trials ordered by total flow (low -> high)")
    axB.set_ylabel("per-trial MAE")
    axB.set_title("(B) Per-trial MAE vs total flow")
    axB.legend()
    axB.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    out_dir = os.path.dirname(metrics_output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(metrics_output_path, dpi=200)
    print(f"Metrics plot saved to {metrics_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate + compare synthetic vs physical DCS g2")
    parser.add_argument("--pathlengths", type=str, default="data/pathlengths.npz")
    parser.add_argument("--output-plot", type=str, default="plots/g2_comparison.png")
    parser.add_argument("--metrics-plot", type=str, default="plots/g2_metrics.png")
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-deep-zero", action="store_true")
    args = parser.parse_args()

    run_comparison(args.pathlengths, args.output_plot, metrics_output_path=args.metrics_plot,
                    test_frac=args.test_frac, seed=args.seed,
                    skip_deep_flow_zero=not args.include_deep_zero)


if __name__ == "__main__":
    main()
