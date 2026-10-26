"""
Compare module: calibrates the single flow->D_B scale factor `a` (and the
zero-flow baseline D_B0) against the PHYSICAL lab g2 curves in core/data/
(the ground truth chosen on 2026-09-30, over the paper's published c1/c2/c3
equations -- those are used only as the secondary geometry gate in
bfi_fit.py), then reports held-out R^2 / MAE / RMSE per SDS.

Ported from core/code/compare.py, with:
  - bug 3.6 preserved as a fix: ONE shared scale `a` for both layers, not two
    independent scale_s/scale_d (that was overfitting, see analysis doc 3.6).
  - the (0,0) condition is excluded from the train/test combos by default, per
    Goal.md (clarified 2026-10-15: deep == 0 with shallow > 0 is kept).
  - Train/test split by PRESET pump combo (132 combos), while the physics is
    driven by the MEASURED outlet flow. Splitting on measured flow (which is
    noisy and near-unique per trial) put repeats of the same condition on both
    sides of the split and made calibration fit ~740 "combos".
  - R^2 is the headline metric (Goal.md's "RMSE ~= 0.95" meant R^2 ~= 0.95).
    Two variants: pooled over all (tau, trial) points, which penalises errors
    that are systematic across flow conditions, and the median per-trial R^2,
    i.e. how well a typical single curve is reproduced.
  - BFI agreement: sim vs physical test curves fitted with the SAME
    semi-infinite fitter (bfi_fit.py), reported as Pearson R and median ratio.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parameters as cfg
import bfi_fit
import data_io
from dcs_physics import calculate_g2

CAL_TAU_STRIDE = 5    # calibrate on every 5th tau point (101 of 501) for speed
BFI_MM2S_TO_FILE = 1e7   # fitter returns mm^2/s; metadata.csv BFI is in 1e-9 cm^2/s


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


def _flow_bin_breakdown(sim, ref, fr_s, fr_d):
    """Per-trial R^2 (median) and pooled RMSE for low/high shallow x low/high deep
    flow, split at the median FR_meas of each layer."""
    s_hi, d_hi = fr_s > np.median(fr_s), fr_d > np.median(fr_d)
    out = {}
    for sname, smask in (("shallow-low", ~s_hi), ("shallow-high", s_hi)):
        for dname, dmask in (("deep-low", ~d_hi), ("deep-high", d_hi)):
            idx = np.where(smask & dmask)[0]
            if len(idx) == 0:
                continue
            out[f"{sname}/{dname}"] = {
                "n": int(len(idx)),
                "r2_per_trial_median": [float(np.median([_r2(sim[s][:, j], ref[s][:, j]) for j in idx]))
                                        for s in range(sim.shape[0])],
                "rmse": [float(np.sqrt(np.mean((sim[s][:, idx] - ref[s][:, idx]) ** 2)))
                         for s in range(sim.shape[0])],
            }
    return out


def _r2(sim, ref):
    """R^2 = 1 - SS_res/SS_tot over all points of the given arrays."""
    ss_res = np.sum((ref - sim) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def run_comparison(pathlengths_path, plot_output_path, metrics_output_path="plots/g2_metrics.png",
                    test_frac=0.30, seed=0, cal_max_photons=5000,
                    skip_deep_flow_zero=cfg.SKIP_DEEP_FLOW_ZERO,
                    metrics_json_path="plots/metrics.json"):
    meta = data_io.load_physical_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    tau = data_io.TAU_ARRAY
    fr_s, fr_d = meta["fr_meas_s"], meta["fr_meas_d"]

    keep = np.ones(meta["n_trials"], dtype=bool)
    if skip_deep_flow_zero:
        keep = ~data_io.excluded_condition(meta["fr_set_s"], meta["fr_set_d"])
        print(f"Excluding {np.sum(~keep)} physical trials of the (0,0) condition "
              f"({np.sum(keep)} remain), per Goal.md")

    pl = np.load(pathlengths_path)
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, cal_max_photons, rng) for s in range(3)}

    kept_idx = np.where(keep)[0]
    train_sub, test_sub, n_combos = _combo_split(meta["fr_set_s"][kept_idx],
                                                  meta["fr_set_d"][kept_idx], test_frac, seed)
    train_idx, test_idx = kept_idx[train_sub], kept_idx[test_sub]
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train/test overlap!"
    print(f"{n_combos} preset flow combos -> {len(train_idx)} train / {len(test_idx)} test trials")

    # --- Calibrate (a, D_B0) jointly, one representative trial per TRAIN preset combo ---
    seen, cal_trials = set(), []
    for i in train_idx:
        key = (meta["fr_set_s"][i], meta["fr_set_d"][i])
        if key not in seen:
            seen.add(key)
            cal_trials.append(i)
    cal_trials = np.array(cal_trials)
    tau_cal = tau[::CAL_TAU_STRIDE]
    ref_cal = phys_g2[:, ::CAL_TAU_STRIDE, :][:, :, cal_trials]

    def cal_loss(log_params):
        a, db0 = 10.0 ** log_params[0], 10.0 ** log_params[1]
        sim = _synth_curves(det, fr_s, fr_d, beta, tau_cal, cal_trials, a, db0)
        return np.mean((sim - ref_cal) ** 2)

    print(f"Calibrating (a, D_B0) on {len(cal_trials)} train combos...")
    x0 = [np.log10(cfg.DEFAULT_A_MM2_S_PER_FR), np.log10(cfg.DEFAULT_DB0_MM2_S)]
    res = minimize(cal_loss, x0, method="Nelder-Mead",
                    bounds=[(-9.0, -3.0), (-9.0, -5.0)])
    a_star, db0_star = 10.0 ** res.x[0], 10.0 ** res.x[1]
    print(f"  calibrated a={a_star:.4e} mm^2/s per (ml/min/100g)   D_B0={db0_star:.4e} mm^2/s "
          f"({res.nfev} loss evaluations, converged={res.success})")

    # --- Report metrics on TEST trials only ---
    sim_test = _synth_curves(det, fr_s, fr_d, beta, tau, test_idx, a_star, db0_star)
    ref_test = phys_g2[:, :, test_idx]
    print("Fitting BFI (semi-infinite) on sim and physical TEST curves...")
    bfi_sim = bfi_fit.fit_bfi_all(sim_test, beta, tau)
    bfi_phys = bfi_fit.fit_bfi_all(ref_test, beta, tau)

    # Reference definitions (see plan 2026-10-07):
    #   input to sim   = FR_meas (what actually flowed)
    #   g2 reference   = physical g2 of the SAME trial
    #   split grouping = FR_set preset combo
    #   BFI-X          = sim BFI vs physical-g2 BFI, same fitter (isolates sim error)
    #   BFI-Y          = sim BFI vs BFI column in metadata.csv (lab/paper fitter; the
    #                    absolute scale differs per SDS, see bfi_fit_vs_file_ratio)
    bfi_file = meta["bfi"][:, test_idx]            # 1e-9 cm^2/s
    bfi_sim_file_units = bfi_sim * BFI_MM2S_TO_FILE
    bfi_phys_file_units = bfi_phys * BFI_MM2S_TO_FILE

    per_sds = []
    per_trial_mae = np.zeros((3, len(test_idx)))
    print("\n================= Held-out (TEST) metrics =================")
    for s in range(3):
        diff = sim_test[s] - ref_test[s]
        per_trial_r2 = [_r2(sim_test[s][:, j], ref_test[s][:, j]) for j in range(len(test_idx))]
        per_trial_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
        per_trial_mae[s] = np.mean(np.abs(diff), axis=0)
        row = {
            "sds": data_io.SDS_LABELS[s],
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "rmse_per_trial_median": float(np.median(per_trial_rmse)),
            "r2_per_trial_median": float(np.median(per_trial_r2)),
            "r2_pooled": float(_r2(sim_test[s], ref_test[s])),
            "bfi_X_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_phys[s])[0, 1]),
            "bfi_X_median_ratio_sim_over_phys": float(np.median(bfi_sim[s] / bfi_phys[s])),
            "bfi_X_rmse_1e-9cm2s": float(np.sqrt(np.mean((bfi_sim_file_units[s] - bfi_phys_file_units[s]) ** 2))),
            "bfi_Y_pearson_r": float(np.corrcoef(bfi_sim_file_units[s], bfi_file[s])[0, 1]),
            "bfi_Y_median_ratio_sim_over_file": float(np.median(bfi_sim_file_units[s] / bfi_file[s])),
            "bfi_Y_rmse_1e-9cm2s": float(np.sqrt(np.mean((bfi_sim_file_units[s] - bfi_file[s]) ** 2))),
            "bfi_fit_vs_file_ratio": float(np.median(bfi_phys_file_units[s] / bfi_file[s])),
        }
        per_sds.append(row)

    print(f"{'SDS':<5}{'RMSE':>9}{'RMSE/trial':>12}{'R2/trial':>10}{'R2 pooled':>11}"
          f"{'BFI-X R':>10}{'X ratio':>9}{'BFI-Y R':>10}{'Y ratio':>9}")
    for r in per_sds:
        print(f"{r['sds']:<5}{r['rmse']:>9.4f}{r['rmse_per_trial_median']:>12.4f}"
              f"{r['r2_per_trial_median']:>10.4f}{r['r2_pooled']:>11.4f}"
              f"{r['bfi_X_pearson_r']:>10.4f}{r['bfi_X_median_ratio_sim_over_phys']:>9.3f}"
              f"{r['bfi_Y_pearson_r']:>10.4f}{r['bfi_Y_median_ratio_sim_over_file']:>9.3f}")
    print("  (target: R^2 >= 0.95)   BFI-X = sim vs physical g2, same fitter;  "
          "BFI-Y = sim vs BFI column in metadata.csv")
    print("  our fitter on PHYSICAL g2 vs file BFI (should be ~1 if fitters agree): "
          + "  ".join(f"{r['sds']}={r['bfi_fit_vs_file_ratio']:.3f}" for r in per_sds))

    flow_bins = _flow_bin_breakdown(sim_test, ref_test, fr_s[test_idx], fr_d[test_idx])
    print("\nBy flow condition (test trials; low/high split at median of each layer's FR_meas):")
    for name, vals in flow_bins.items():
        print(f"  {name:<22} n={vals['n']:<4} " + "  ".join(
            f"{lbl}: R2={vals['r2_per_trial_median'][k]:.3f} RMSE={vals['rmse'][k]:.4f}"
            for k, lbl in enumerate(data_io.SDS_LABELS)))
    print("===========================================================")

    metrics = {"a_mm2_s_per_fr": a_star, "db0_mm2_s": db0_star,
               "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
               "n_preset_combos": int(n_combos), "cal_photons_per_sds": cal_max_photons,
               "definitions": {"sim_input": "FR_meas", "g2_reference": "physical g2, same trial",
                               "split_group": "FR_set preset combo",
                               "bfi_units": "1e-9 cm^2/s"},
               "per_sds": per_sds, "by_flow_condition": flow_bins}
    os.makedirs(os.path.dirname(metrics_json_path) or ".", exist_ok=True)
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_json_path}")

    r2 = np.array([r["r2_pooled"] for r in per_sds])
    _overlay_plot(det, fr_s, fr_d, beta, tau, phys_g2, test_idx, a_star, db0_star, plot_output_path)
    _metrics_plot(r2, per_trial_mae, fr_s[test_idx] + fr_d[test_idx], metrics_output_path)
    return metrics


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


def _metrics_plot(r2, per_trial_mae, total_flow, metrics_output_path):
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
