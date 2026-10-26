"""
Stratified-subset experiment (plan 2026-10-07): evaluate the pipeline on a small
set of flow conditions that still covers the whole (shallow x deep) space,
instead of all 1,090 trials. Reuses the existing data/pathlengths.npz (no MCX).

  E1  coverage      : 4x4 flow grid, per cell 2 train combos / 1 held-out test combo
  E2  calib size    : same test set, calibrate on 8 / 16 / 32 stratified train combos
  E3  extrapolation : train on low-mid flow only, test on high-high and on the
                      shallow-low/deep-high corner (the known weak case)

Rules inherited from src/compare.py: sim is driven by FR_meas, reference is the
physical g2 of the SAME trial, grouping/splitting is by FR_set preset combo, and
deep-flow==0 (hence also (0,0)) is excluded.

Usage:  .venv/bin/python scripts/subset_experiment.py [--seeds 0 1 2] [--per-combo 6]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
from config import parameters as cfg
import bfi_fit
import data_io
import compare
from compare import _synth_curves, _r2, CAL_TAU_STRIDE, BFI_MM2S_TO_FILE

N_BINS = 4
SDS = data_io.SDS_LABELS


# ---------------------------------------------------------------- design
def build_design(meta, per_combo, seed):
    """Group non-excluded trials by FR_set combo, cap trials per combo, and assign
    every combo to a cell of the 4x4 (shallow bin, deep bin) grid by rank quantile."""
    keep = np.where(~data_io.excluded_condition(meta["fr_set_s"], meta["fr_set_d"]))[0]
    rng = np.random.default_rng(seed)
    combos = {}
    for i in keep:
        combos.setdefault((meta["fr_set_s"][i], meta["fr_set_d"][i]), []).append(i)
    keys = sorted(combos)
    s_val = np.array([k[0] for k in keys])
    d_val = np.array([k[1] for k in keys])
    s_bin = np.minimum(N_BINS - 1, ((rankdata(s_val) - 1) / len(keys) * N_BINS).astype(int))
    d_bin = np.minimum(N_BINS - 1, ((rankdata(d_val) - 1) / len(keys) * N_BINS).astype(int))
    out = []
    for k, sb, db in zip(keys, s_bin, d_bin):
        idx = np.array(combos[k])
        if len(idx) > per_combo:
            idx = np.sort(rng.choice(idx, per_combo, replace=False))
        out.append({"key": k, "sb": int(sb), "db": int(db), "trials": idx})
    return out


def pick_cells(combos, per_cell=3):
    """Per cell keep up to `per_cell` combos spread over total flow (low/median/high)."""
    cells = {}
    for c in combos:
        cells.setdefault((c["sb"], c["db"]), []).append(c)
    sel = {}
    for cell, lst in cells.items():
        lst = sorted(lst, key=lambda c: c["key"][0] + c["key"][1])
        if len(lst) > per_cell:
            pos = np.linspace(0, len(lst) - 1, per_cell).round().astype(int)
            lst = [lst[p] for p in pos]
        sel[cell] = lst
    return sel


def split_e1(sel, seed):
    """Per cell: 3 combos -> test = sorted[1,0,2][seed%3], rest train; 2 -> 1/1; 1 -> train only."""
    train, test = [], []
    for cell, lst in sel.items():
        if len(lst) == 1:
            train += lst
        elif len(lst) == 2:
            train.append(lst[0]); test.append(lst[1])
        else:
            t = [1, 0, 2][seed % 3]
            test.append(lst[t]); train += [c for j, c in enumerate(lst) if j != t]
    return train, test


def stratified_subset(train, n):
    """Round-robin over cells so a small calibration set still spans the grid."""
    by_cell = {}
    for c in train:
        by_cell.setdefault((c["sb"], c["db"]), []).append(c)
    out, cells = [], sorted(by_cell)
    r = 0
    while len(out) < min(n, len(train)):
        for cell in cells:
            if r < len(by_cell[cell]) and len(out) < n:
                out.append(by_cell[cell][r])
        r += 1
    return out


# ------------------------------------------------------------- physics
def calibrate(det, fr_s, fr_d, beta, tau, phys_g2, train_combos):
    cal_trials = np.array([c["trials"][0] for c in train_combos])
    tau_cal = tau[::CAL_TAU_STRIDE]
    ref = phys_g2[:, ::CAL_TAU_STRIDE, :][:, :, cal_trials]

    def loss(p):
        sim = _synth_curves(det, fr_s, fr_d, beta, tau_cal, cal_trials, 10.0 ** p[0], 10.0 ** p[1])
        return np.mean((sim - ref) ** 2)

    x0 = [np.log10(cfg.DEFAULT_A_MM2_S_PER_FR), np.log10(cfg.DEFAULT_DB0_MM2_S)]
    res = minimize(loss, x0, method="Nelder-Mead", bounds=[(-9.0, -3.0), (-9.0, -5.0)])
    return 10.0 ** res.x[0], 10.0 ** res.x[1]


class Evaluator:
    """Caches the BFI fit of the physical test curves (independent of calibration)."""
    def __init__(self, det, meta, phys_g2, beta, tau):
        self.det, self.meta, self.g2, self.beta, self.tau = det, meta, phys_g2, beta, tau
        self._phys_bfi = {}

    def phys_bfi(self, idx):
        for i in idx:
            if i not in self._phys_bfi:
                self._phys_bfi[i] = bfi_fit.fit_bfi_all(self.g2[:, :, [i]], self.beta, self.tau)[:, 0]
        return np.array([self._phys_bfi[i] for i in idx]).T * BFI_MM2S_TO_FILE   # (3, n)

    def run(self, idx, a, db0):
        idx = np.asarray(idx)
        sim = _synth_curves(self.det, self.meta["fr_meas_s"], self.meta["fr_meas_d"],
                            self.beta, self.tau, idx, a, db0)
        ref = self.g2[:, :, idx]
        bfi_sim = bfi_fit.fit_bfi_all(sim, self.beta, self.tau) * BFI_MM2S_TO_FILE
        bfi_phys = self.phys_bfi(idx)
        bfi_file = self.meta["bfi"][:, idx]
        rows = []
        for s in range(3):
            d = sim[s] - ref[s]
            r2t = [_r2(sim[s][:, j], ref[s][:, j]) for j in range(len(idx))]
            rows.append({
                "sds": SDS[s], "n": int(len(idx)),
                "rmse": float(np.sqrt(np.mean(d ** 2))),
                "r2_per_trial_median": float(np.median(r2t)),
                "r2_pooled": float(_r2(sim[s], ref[s])),
                "bfi_X_pearson_r": _corr(bfi_sim[s], bfi_phys[s]),
                "bfi_X_ratio": float(np.median(bfi_sim[s] / bfi_phys[s])),
                "bfi_Y_pearson_r": _corr(bfi_sim[s], bfi_file[s]),
                "bfi_Y_ratio": float(np.median(bfi_sim[s] / bfi_file[s])),
            })
        return rows


def _corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def trials_of(combos):
    return np.concatenate([c["trials"] for c in combos]) if combos else np.array([], int)


def print_rows(title, rows):
    print(f"  {title}")
    for r in rows:
        print(f"    {r['sds']}: n={r['n']:<4} RMSE={r['rmse']:.4f}  R2/trial={r['r2_per_trial_median']:.3f}  "
              f"R2pool={r['r2_pooled']:.3f}  BFI-X R={r['bfi_X_pearson_r']:.3f} ratio={r['bfi_X_ratio']:.2f}  "
              f"BFI-Y R={r['bfi_Y_pearson_r']:.3f}")


# ------------------------------------------------------------------ main
def run_seed(seed, args, ctx):
    meta, det, phys_g2, beta, tau, ev = ctx
    fr_s, fr_d = meta["fr_meas_s"], meta["fr_meas_d"]
    combos = build_design(meta, args.per_combo, seed)
    sel = pick_cells(combos, args.per_cell)
    n_sel = sum(len(v) for v in sel.values())
    train, test = split_e1(sel, seed)
    assert not ({c["key"] for c in train} & {c["key"] for c in test}), "train/test combo overlap"
    print(f"\n##### seed {seed}: {len(combos)} combos -> {len(sel)} cells, {n_sel} selected combos "
          f"({len(train)} train / {len(test)} test), {len(trials_of(train))}+{len(trials_of(test))} trials")
    out = {"seed": seed, "n_cells": len(sel), "n_train_combos": len(train), "n_test_combos": len(test),
           "n_test_trials": int(len(trials_of(test)))}

    # E1
    a, db0 = calibrate(det, fr_s, fr_d, beta, tau, phys_g2, train)
    rows = ev.run(trials_of(test), a, db0)
    print(f"E1 coverage  (a={a:.3e}, D_B0={db0:.3e})")
    print_rows("overall held-out", rows)
    cell_rows = {}
    for c in test:
        r = ev.run(c["trials"], a, db0)
        cell_rows[f"{c['sb']},{c['db']}"] = {"key": list(map(float, c["key"])), "rows": r}
    out["E1"] = {"a": a, "db0": db0, "overall": rows, "cells": cell_rows}

    # E2
    out["E2"] = {}
    for n in args.calib_sizes:
        sub = stratified_subset(train, n)
        a2, d2 = calibrate(det, fr_s, fr_d, beta, tau, phys_g2, sub)
        r = ev.run(trials_of(test), a2, d2)
        out["E2"][str(len(sub))] = {"a": a2, "db0": d2, "rows": r}
        print(f"E2 calib size {len(sub)} combos (a={a2:.3e}, D_B0={d2:.3e}):  "
              + "  ".join(f"{x['sds']} R2={x['r2_per_trial_median']:.3f} RMSE={x['rmse']:.4f}" for x in r))

    # E3
    low = [c for cs in sel.values() for c in cs if c["sb"] <= 1 and c["db"] <= 1]
    high = [c for cs in sel.values() for c in cs if c["sb"] >= 2 and c["db"] >= 2]
    corner = [c for cs in sel.values() for c in cs if c["sb"] == 0 and c["db"] == N_BINS - 1]
    a3, d3 = calibrate(det, fr_s, fr_d, beta, tau, phys_g2, low)
    out["E3"] = {"n_train_combos": len(low), "a": a3, "db0": d3}
    print(f"E3 extrapolation  (train: {len(low)} low-mid combos, a={a3:.3e}, D_B0={d3:.3e})")
    for name, tset in (("high-high", high), ("shallow-low/deep-high corner", corner)):
        r = ev.run(trials_of(tset), a3, d3)
        out["E3"][name] = {"n_combos": len(tset), "rows": r}
        print_rows(f"test on {name} ({len(tset)} combos)", r)
    # reference: same corner with the full E1 calibration (is the corner a coverage problem?)
    if corner:
        r = ev.run(trials_of(corner), a, db0)
        out["E3"]["corner_with_E1_calibration"] = {"rows": r}
        print_rows("same corner using E1 calibration (covers the grid)", r)
    return out


def make_plots(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    r0 = results[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for s in range(3):
        for k, (metric, title, cmap) in enumerate((("r2_per_trial_median", "R2 / trial (median)", "RdYlGn"),
                                                    ("rmse", "RMSE", "RdYlGn_r"))):
            grid = np.full((N_BINS, N_BINS), np.nan)
            for key, v in r0["E1"]["cells"].items():
                sb, db = map(int, key.split(","))
                grid[sb, db] = v["rows"][s][metric]
            ax = axes[k, s]
            im = ax.imshow(grid, origin="lower", cmap=cmap,
                           vmin=0.5 if k == 0 else 0.0, vmax=1.0 if k == 0 else 0.1)
            for sb in range(N_BINS):
                for db in range(N_BINS):
                    if not np.isnan(grid[sb, db]):
                        ax.text(db, sb, f"{grid[sb, db]:.2f}" if k == 0 else f"{grid[sb, db]:.3f}",
                                ha="center", va="center", fontsize=8)
            ax.set_xlabel("deep-flow bin (low->high)")
            ax.set_ylabel("shallow-flow bin (low->high)")
            ax.set_title(f"{SDS[s]} SDS: {title}")
            fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "subset_heatmap.png"), dpi=150)
    plt.close(fig)

    sizes = sorted(int(n) for n in results[0]["E2"])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))
    for s, col in zip(range(3), ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        r2 = np.array([[r["E2"][str(n)]["rows"][s]["r2_per_trial_median"] for n in sizes] for r in results])
        rm = np.array([[r["E2"][str(n)]["rows"][s]["rmse"] for n in sizes] for r in results])
        axA.errorbar(sizes, r2.mean(0), yerr=[r2.mean(0) - r2.min(0), r2.max(0) - r2.mean(0)],
                     color=col, marker="o", capsize=3, label=SDS[s])
        axB.errorbar(sizes, rm.mean(0), yerr=[rm.mean(0) - rm.min(0), rm.max(0) - rm.mean(0)],
                     color=col, marker="o", capsize=3, label=SDS[s])
    axA.axhline(0.95, color="k", ls="--", lw=1)
    axA.set_xlabel("# calibration combos"); axA.set_ylabel("R2 / trial (median)"); axA.legend(); axA.grid(ls=":")
    axB.set_xlabel("# calibration combos"); axB.set_ylabel("RMSE"); axB.grid(ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "subset_calib_size.png"), dpi=150)
    plt.close(fig)


def summarize(results, baseline):
    print("\n================ SUMMARY over seeds (E1 held-out, mean [min,max]) ================")
    print(f"{'SDS':<5}{'RMSE':>20}{'R2/trial':>24}{'BFI-X R':>22}{'full-data R2/trial':>20}")
    for s in range(3):
        def stat(k):
            v = np.array([r["E1"]["overall"][s][k] for r in results])
            return f"{v.mean():.3f} [{v.min():.3f},{v.max():.3f}]"
        b = baseline[s]["r2_per_trial_median"] if baseline else float("nan")
        print(f"{SDS[s]:<5}{stat('rmse'):>20}{stat('r2_per_trial_median'):>24}{stat('bfi_X_pearson_r'):>22}{b:>20.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-combo", type=int, default=6)
    ap.add_argument("--per-cell", type=int, default=3)
    ap.add_argument("--calib-sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--pathlengths", default=os.path.join(ROOT, "data", "pathlengths.npz"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "plots"))
    args = ap.parse_args()

    t0 = time.time()
    meta = data_io.load_physical_metadata()
    phys_g2 = data_io.load_physical_g2()
    beta = data_io.extract_beta(phys_g2)
    tau = data_io.TAU_ARRAY
    pl = np.load(args.pathlengths)
    rng = np.random.default_rng(42)
    det = {s: data_io.downsample_detector(pl, s + 1, 5000, rng) for s in range(3)}
    ctx = (meta, det, phys_g2, beta, tau, Evaluator(det, meta, phys_g2, beta, tau))

    results = [run_seed(s, args, ctx) for s in args.seeds]

    baseline = None
    bpath = os.path.join(args.outdir, "metrics.json")
    if os.path.exists(bpath):
        baseline = json.load(open(bpath))["per_sds"]
    summarize(results, baseline)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "subset_experiment.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    make_plots(results, args.outdir)
    print(f"\nDone in {time.time() - t0:.0f}s -> {args.outdir}/subset_experiment.json, subset_heatmap.png, subset_calib_size.png")


if __name__ == "__main__":
    main()
