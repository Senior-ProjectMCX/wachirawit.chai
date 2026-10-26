"""
Our model vs Wichuda_r's model on the closest-row test set (dataset/testset_closest.csv,
132 trials, one per condition, (0,0) excluded), scored with Compare Method (Simple):
RMSE and R^2 of g2 vs the physical g2 of the same trial, BFI-X and BFI-Y.

Cases
  ours           our MCX photons, FR_meas, (a, D_B0) calibrated on the test set, beta per SDS
                 (= src/compare_simple.py --testset)
  wichuda as-is  Wichuda's photons and rules exactly as her code runs them
                 (../wichuda_g2): preset flow, alpha*D_B = 1e-6 * flow, no D_B0,
                 beta per condition from a double-exponential fit of that condition's
                 physical trials
  wichuda + our rules   Wichuda's photons, but our rules (FR_meas, calibrated a/D_B0,
                 beta per SDS) -> isolates the effect of her MCX photon data/geometry

Usage: .venv/bin/python scripts/compare_with_wichuda.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "wichuda_g2"))
from config import parameters as cfg
import bfi_fit
import data_io
from compare import _synth_curves, _r2, CAL_TAU_STRIDE, BFI_MM2S_TO_FILE
import wichuda_g2 as wg


def calibrate(det, fr_s, fr_d, beta, tau, g2, trials):
    tc = tau[::CAL_TAU_STRIDE]
    ref = g2[:, ::CAL_TAU_STRIDE, :][:, :, trials]

    def loss(p):
        return np.mean((_synth_curves(det, fr_s, fr_d, beta, tc, trials, 10 ** p[0], 10 ** p[1]) - ref) ** 2)

    r = minimize(loss, [np.log10(cfg.DEFAULT_A_MM2_S_PER_FR), np.log10(cfg.DEFAULT_DB0_MM2_S)],
                 method="Nelder-Mead", bounds=[(-9.0, -3.0), (-9.0, -5.0)])
    return 10 ** r.x[0], 10 ** r.x[1]


def score(name, sim, ref, beta, tau, bfi_phys, bfi_file):
    bfi_sim = bfi_fit.fit_bfi_all(sim, beta, tau) * BFI_MM2S_TO_FILE
    rows = []
    for s in range(3):
        d = sim[s] - ref[s]
        rows.append({
            "sds": data_io.SDS_LABELS[s],
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "r2_per_trial_median": float(np.median([_r2(sim[s][:, j], ref[s][:, j]) for j in range(ref.shape[2])])),
            "r2_pooled": float(_r2(sim[s], ref[s])),
            "bfi_X_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_phys[s])[0, 1]),
            "bfi_X_median_ratio": float(np.median(bfi_sim[s] / bfi_phys[s])),
            "bfi_Y_pearson_r": float(np.corrcoef(bfi_sim[s], bfi_file[s])[0, 1]),
            "bfi_Y_median_ratio": float(np.median(bfi_sim[s] / bfi_file[s])),
        })
    per_trial = {
        "rmse": np.sqrt(np.mean((sim - ref) ** 2, axis=1)),                      # (3, n)
        "r2": np.array([[_r2(sim[s][:, j], ref[s][:, j]) for j in range(ref.shape[2])] for s in range(3)]),
        "bfi_abs_err": np.abs(bfi_sim - bfi_phys),                              # BFI-X, per trial
        "bfi": bfi_sim,
    }
    print(f"\n[{name}]")
    for r in rows:
        print(f"  {r['sds']}: RMSE={r['rmse']:.4f}  R2/trial={r['r2_per_trial_median']:.3f}  "
              f"R2pool={r['r2_pooled']:.3f}  BFI-X R={r['bfi_X_pearson_r']:.3f} "
              f"ratio={r['bfi_X_median_ratio']:.2f}  BFI-Y R={r['bfi_Y_pearson_r']:.3f}", flush=True)
    return rows, per_trial


def head_to_head(per_trial, trials, meta, bfi_phys):
    """Per-trial, per-SDS comparison (no averaging of RMSE across trials): for every test
    trial and SDS, the case with the lower RMSE / higher R^2 / smaller |BFI - BFI_phys| wins.
    Also writes every per-trial value to plots/compare_with_wichuda_per_trial.csv."""
    rows = []
    for j, i in enumerate(trials):
        for s in range(3):
            row = {"Trial_Index": int(i), "preset_shallow": meta["fr_set_s"][i],
                   "preset_deep": meta["fr_set_d"][i], "sds": data_io.SDS_LABELS[s],
                   "bfi_phys": bfi_phys[s, j]}
            for case, pt in per_trial.items():
                row[f"rmse_{case}"] = pt["rmse"][s, j]
                row[f"r2_{case}"] = pt["r2"][s, j]
                row[f"bfi_{case}"] = pt["bfi"][s, j]
                row[f"bfi_abs_err_{case}"] = pt["bfi_abs_err"][s, j]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "plots", "compare_with_wichuda_per_trial.csv"), index=False)

    result = {}
    for other in ("wichuda_as_is", "wichuda_photons_our_rules"):
        print(f"\n===== head-to-head per trial: ours vs {other} (n = {len(trials)} trials per SDS) =====")
        res = {}
        for metric, lower_better in (("rmse", True), ("r2", False), ("bfi_abs_err", True)):
            per_sds = {}
            for sds in data_io.SDS_LABELS:
                d = df[df["sds"] == sds]
                o, w = d[f"{metric}_ours"].to_numpy(), d[f"{metric}_{other}"].to_numpy()
                ours_win = (o < w) if lower_better else (o > w)
                tie = np.isclose(o, w, rtol=0, atol=1e-12)
                ours_win &= ~tie
                per_sds[sds] = {"ours": int(ours_win.sum()), "other": int((~ours_win & ~tie).sum()),
                                "tie": int(tie.sum())}
            tot = {k: sum(v[k] for v in per_sds.values()) for k in ("ours", "other", "tie")}
            res[metric] = {"per_sds": per_sds, "total": tot}
            print(f"  {metric:<11} " + "  ".join(
                f"{sds}: ours {v['ours']:>3} - {v['other']:<3}" for sds, v in per_sds.items())
                  + f"  | total ours {tot['ours']} - {tot['other']} (ties {tot['tie']})")
        # trial-level: who wins more of the 3 SDS on RMSE (no averaging)
        piv_o = df.pivot(index="Trial_Index", columns="sds", values="rmse_ours")
        piv_w = df.pivot(index="Trial_Index", columns="sds", values=f"rmse_{other}")
        sds_wins = (piv_o < piv_w).sum(axis=1)
        res["trial_majority_rmse"] = {"ours": int((sds_wins >= 2).sum()), "other": int((sds_wins <= 1).sum()),
                                      "ours_all_3": int((sds_wins == 3).sum()),
                                      "other_all_3": int((sds_wins == 0).sum())}
        t = res["trial_majority_rmse"]
        print(f"  per trial (wins >= 2 of 3 SDS on RMSE): ours {t['ours']} - {t['other']}  "
              f"(ours all 3: {t['ours_all_3']}, {other} all 3: {t['other_all_3']})")
        result[other] = res
    return result


def main():
    meta = data_io.load_physical_metadata()
    g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(g2)
    tau = data_io.TAU_ARRAY
    trials = np.sort(pd.read_csv(os.path.join(ROOT, "dataset", "testset_closest.csv"))["Trial_Index"].to_numpy(int))
    assert not np.any(data_io.excluded_condition(meta["fr_set_s"][trials], meta["fr_set_d"][trials]))
    ref = g2[:, :, trials]
    print(f"test set: {len(trials)} trials")

    bfi_phys = bfi_fit.fit_bfi_all(ref, beta, tau) * BFI_MM2S_TO_FILE
    bfi_file = meta["bfi"][:, trials]
    out = {"n_trials": int(len(trials)), "cases": {}}

    # 1. ours
    pl = np.load(os.path.join(ROOT, "data", "pathlengths.npz"))
    drng = np.random.default_rng(42)
    det_ours = {s: data_io.downsample_detector(pl, s + 1, 5000, drng) for s in range(3)}
    a, db0 = calibrate(det_ours, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, g2, trials)
    sim = _synth_curves(det_ours, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a, db0)
    rows, pt = score("ours", sim, ref, beta, tau, bfi_phys, bfi_file)
    per_trial = {"ours": pt}
    out["cases"]["ours"] = {"a": a, "db0": db0, "rows": rows}

    # 2. wichuda as-is
    detp = wg.load_detp()
    sim = np.zeros_like(ref)
    beta_cache = {}
    for j, i in enumerate(trials):
        key = (meta["fr_set_s"][i], meta["fr_set_d"][i])
        if key not in beta_cache:
            same = np.where((meta["fr_set_s"] == key[0]) & (meta["fr_set_d"] == key[1]))[0]
            beta_cache[key] = [wg.beta_from_physical_trials(g2[s][:, same].T) for s in range(3)]
        sim[:, :, j] = wg.generate_g2(key[0], key[1], beta_cache[key], detp=detp, tau_seconds=tau)
    rows, per_trial["wichuda_as_is"] = score("wichuda as-is", sim, ref, beta, tau, bfi_phys, bfi_file)
    out["cases"]["wichuda_as_is"] = {"a": wg.FLOW_TO_ALPHA_DB_SCALE, "db0": 0.0, "rows": rows}

    # 3. wichuda photons + our rules
    ph = wg.extract_photon_data(detp)
    det_w = {}
    for s in range(3):
        m = ph["detector_id"] == s + 1
        det_w[s] = (ph["path_shallow"][m].astype(float), ph["path_deep"][m].astype(float),
                    ph["momentum_shallow"][m].astype(float), ph["momentum_deep"][m].astype(float))
    a_w, db0_w = calibrate(det_w, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, g2, trials)
    sim = _synth_curves(det_w, meta["fr_meas_s"], meta["fr_meas_d"], beta, tau, trials, a_w, db0_w)
    rows, per_trial["wichuda_photons_our_rules"] = score("wichuda photons + our rules", sim, ref, beta, tau,
                                                        bfi_phys, bfi_file)
    out["cases"]["wichuda_photons_our_rules"] = {"a": a_w, "db0": db0_w, "rows": rows}

    out["head_to_head"] = head_to_head(per_trial, trials, meta, bfi_phys)

    print(f"\ncalibrated (a, D_B0): ours {a:.3e}, {db0:.3e} | wichuda photons {a_w:.3e}, {db0_w:.3e}")
    path = os.path.join(ROOT, "plots", "compare_with_wichuda.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
