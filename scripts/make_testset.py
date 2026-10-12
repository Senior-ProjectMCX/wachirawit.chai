"""
Build a small test set: for each flow condition (preset pump combo), keep only the ONE
trial whose measured outlet flow is closest to the preset, so a run does not have to
evaluate all 1,540 rows.

  distance = sqrt( (FR_meas_shallow - FR_set_shallow)^2 + (FR_meas_deep - FR_set_deep)^2 )
  ties -> lowest Trial_Index

Deep_Flow_Preset == 0 (which includes (0,0)) is excluded, per Goal.md, leaving one row per
remaining preset combo (108).

Output: dataset/testset_closest.csv (metadata columns + distance + n_repeats in that combo).
Trial_Index is the row index into core/data/{metadata.csv, g2_SDS*.csv}.

Usage: .venv/bin/python scripts/make_testset.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "dataset", "Dataset_metadata.csv")
OUT = os.path.join(ROOT, "dataset", "testset_closest.csv")

SP, DP = "Shallow_Flow_Preset_Pump", "Deep_Flow_Preset_Pump"
SM, DM = "Shallow_Flow_Measured_Outlet", "Deep_Flow_Measured_Outlet"


def main():
    df = pd.read_csv(SRC)
    df = df[df[DP] != 0.0].copy()
    df["distance"] = np.sqrt((df[SM] - df[SP]) ** 2 + (df[DM] - df[DP]) ** 2)
    df["n_repeats"] = df.groupby([SP, DP])[SP].transform("size")
    best = (df.sort_values(["distance", "Trial_Index"])
              .groupby([SP, DP], as_index=False).head(1)
              .sort_values([SP, DP]).reset_index(drop=True))
    best.to_csv(OUT, index=False)

    print(f"{len(best)} conditions -> {OUT}")
    print(f"distance of chosen row: median {best['distance'].median():.3f}, "
          f"max {best['distance'].max():.3f} ml/min/100g")
    print(f"(all rows in these conditions: median {df['distance'].median():.3f})")
    worst = best.nlargest(5, "distance")[["Trial_Index", SP, DP, SM, DM, "distance"]]
    print("conditions whose best row is still far from the preset:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
