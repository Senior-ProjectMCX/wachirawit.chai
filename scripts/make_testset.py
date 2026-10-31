"""
Build a small test set: for each flow condition (preset pump combo), keep only the ONE
trial whose measured outlet flow is closest to the preset, so a run does not have to
evaluate all 1,540 rows.

  distance = sqrt( (FR_meas_shallow - FR_set_shallow)^2 + (FR_meas_deep - FR_set_deep)^2 )
  ties -> lowest Trial_Index

Only the (0,0) condition is excluded, per Goal.md (clarified 2026-10-15; deep == 0 with
shallow > 0 is kept), leaving one row per remaining preset combo (132).

Output: dataset/testset_closest.csv (metadata columns + distance + n_repeats in that combo).
Trial_Index is the row index into core/data/{metadata.csv, g2_SDS*.csv}.

Usage: .venv/bin/python scripts/make_testset.py              (K = 1 -> testset_closest.csv)
       .venv/bin/python scripts/make_testset.py --k 5       (5 closest per condition -> testset_closest_k5.csv,
                                                             with a "rank" column 1..K)
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "dataset", "Dataset_metadata.csv")
OUT = os.path.join(ROOT, "dataset", "testset_closest.csv")

SP, DP = "Shallow_Flow_Preset_Pump", "Deep_Flow_Preset_Pump"
SM, DM = "Shallow_Flow_Measured_Outlet", "Deep_Flow_Measured_Outlet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1, help="closest trials kept per condition")
    args = ap.parse_args()
    out = OUT if args.k == 1 else OUT.replace(".csv", f"_k{args.k}.csv")

    df = pd.read_csv(SRC)
    df = df[~((df[SP] == 0.0) & (df[DP] == 0.0))].copy()
    df["distance"] = np.sqrt((df[SM] - df[SP]) ** 2 + (df[DM] - df[DP]) ** 2)
    df["n_repeats"] = df.groupby([SP, DP])[SP].transform("size")
    ranked = df.sort_values(["distance", "Trial_Index"])
    ranked["rank"] = ranked.groupby([SP, DP]).cumcount() + 1
    best = ranked[ranked["rank"] <= args.k].sort_values([SP, DP, "rank"]).reset_index(drop=True)
    if args.k == 1:
        best = best.drop(columns="rank")
    best.to_csv(out, index=False)

    print(f"{len(best)} rows ({best.groupby([SP, DP]).ngroups} conditions x K = {args.k}) -> {out}")
    print(f"distance of chosen row: median {best['distance'].median():.3f}, "
          f"max {best['distance'].max():.3f} ml/min/100g")
    print(f"(all rows in these conditions: median {df['distance'].median():.3f})")
    worst = best.nlargest(5, "distance")[["Trial_Index", SP, DP, SM, DM, "distance"]]
    print("conditions whose best row is still far from the preset:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
