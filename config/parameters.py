"""
Physical and simulation parameters for the wachirawit.chai DCS two-layer
flow phantom pipeline. Single source of truth -- every module in src/
imports its constants from here instead of redefining them.

Sources:
  - Nakabayashi et al. (2023), Biomed. Opt. Express 14(10) 5358-5375
  - Wu et al. (2021), Neurophotonics 8(1) 015001
  - core/code/{simulation,dynamic_flow,c2c1_test}.py (prior working pipeline,
    see "DCS Two-Layer Flow Phantom Simulation Analysis" in Obsidian for the
    10 known bugs this config deliberately avoids)
"""
import numpy as np

# ============================================================
# OPTICAL PROPERTIES (785 nm phantom fluid, identical in both layers)
# ============================================================
WAVELENGTH_NM = 785.0
WAVELENGTH_MM = WAVELENGTH_NM * 1e-6
REFRACTIVE_INDEX_N = 1.37

MU_A_CM = 0.56
MU_SP_CM = 4.81
MU_A_MM = MU_A_CM / 10.0          # 0.056 mm^-1
MU_SP_MM = MU_SP_CM / 10.0        # 0.481 mm^-1
ANISOTROPY_G = 0.89
MU_S_MM = MU_SP_MM / (1.0 - ANISOTROPY_G)

K0 = 2.0 * np.pi * REFRACTIVE_INDEX_N / WAVELENGTH_MM   # mm^-1

# ============================================================
# PHANTOM GEOMETRY
# ============================================================
VOXEL_SIZE_MM = 0.5
PHANTOM_X_MM = 100.0      # physical phantom outer dimensions ('realistic' geometry)
PHANTOM_Y_MM = 50.0
PHANTOM_Z_MM = 50.5
SHALLOW_DEPTH_MM = 5.5
DEEP_DEPTH_MM = 35.0
TOTAL_DEPTH_MM = SHALLOW_DEPTH_MM + DEEP_DEPTH_MM

# Default geometry = 'simplified' slab (no Mylar voxel layer). Bug 3.2 in the
# analysis doc: the physical 0.05mm Mylar sheet forced into a 0.5mm voxel
# becomes a 10x-too-thick, over-scattering barrier. The slab model matches
# the physical dataset better and is what the authors' own MC setup assumes.
DEFAULT_GEOMETRY = "simplified"

# Slab extent is chosen to approximate a laterally semi-infinite medium: the
# source sits 40mm from the -x face so the 3cm detector is 50mm from +x.
# (The 60x60mm slab ported from core/ centred the source, which put the 3cm
# detector exactly ON the +x domain face -- half its aperture fell outside
# the grid, the real cause of the low SDS=3cm photon counts.)
SLAB_X_MM = 120.0
SLAB_Y_MM = 80.0
SLAB_Z_MM = 50.0
SLAB_SOURCE_X_MM = 40.0

# MCX per-face boundary string, faces ordered -x,-y,-z,+x,+y,+z
# ('a' absorb, 'r' Fresnel reflection, 'm' mirror -- see mcxcl README "--bc").
# Top face (-z, tissue/air, n=1.37 vs 1.0) is Fresnel; the far faces absorb.
MCX_BC = "aaraaa"

# 'paper' geometry: the fluid volume at the paper's stated size (Nakabayashi Sec. 2.1:
# phantom 100 x 50 mm, shallow 5.5 mm + deep 35 mm). Acrylic wall thickness is not
# given, so the fluid fills the full 100 x 50 footprint; Mylar (0.05 mm) is not
# voxelised (bug 3.2). The phantom is a closed box, so every face is Fresnel ('r':
# fluid n=1.37 vs outside n=1.0) instead of the absorbing far faces of the slab.
# Probe placement is not given either: the source-to-3cm-detector span is centred on
# the 100 mm length (source at 35 mm, detectors at 45/55/65 mm), y at the centre.
PAPER_FLUID_Z_MM = SHALLOW_DEPTH_MM + DEEP_DEPTH_MM      # 40.5
PAPER_SOURCE_X_MM = (PHANTOM_X_MM - 30.0) / 2.0          # 35.0
PAPER_BC = "rrrrrr"

# ============================================================
# SOURCE / DETECTOR
# ============================================================
SDS_MM = np.array([10.0, 20.0, 30.0])      # 1, 2, 3 cm
SDS_LABELS = ["1cm", "2cm", "3cm"]
DET_RADIUS_MM = 1.0

# ============================================================
# DCS DELAY TIME AXIS
# ============================================================
N_TAU = 501
TAU_US = np.arange(N_TAU, dtype=float)     # 0, 1, ..., 500 us
TAU_S = TAU_US * 1e-6

# ============================================================
# DYNAMICS: flow -> alpha*D_B
# ============================================================
# alpha*D_B = a * max(FR, 0) + D_B0
#   a    : calibrated scale, SAME for both layers (bug 3.6 -- one physical
#          fluid/bead matrix in both layers means one scale, not two)
#   D_B0 : thermal-Brownian baseline so g2 still decays at FR=0 (bug 3.7)
DEFAULT_A_MM2_S_PER_FR = 1e-6
DEFAULT_DB0_MM2_S = 4.5e-7


def flow_to_alpha_db(flow, a=DEFAULT_A_MM2_S_PER_FR, db0=DEFAULT_DB0_MM2_S):
    """alpha*D_B (mm^2/s) = a * max(flow, 0) + D_B0. Negative measured flow
    (scale noise/evaporation at FR=0, see "Data set definition" note) is
    clamped to zero rather than allowed to subtract from the baseline."""
    return a * np.maximum(flow, 0.0) + db0


# ============================================================
# SEMI-INFINITE HOMOGENEOUS MODEL (BFI fitting / geometry gate)
# ============================================================
Z0_MM = 1.0 / MU_SP_MM
# zb = 1.76/mu_s' for n=1.37, as given by Wu et al. 2021 (Nakabayashi's ref [37]).
# Nakabayashi gives no Reff value, and the zb formula printed in their Eq. 3,
# 2(1-Reff)/(3 mu_s'(1+Reff)), is inverted relative to the standard form.
ZB_MM = 1.76 / MU_SP_MM

# Target layer-sensitivity ratios c2/c1 from Nakabayashi Eq.(10)-(12); the
# simulated geometry must reproduce these BEFORE any flow->D_B calibration.
TARGET_C2_C1 = {0: 0.140, 1: 0.385, 2: 0.459}
TARGET_C1_C2_C3 = {
    0: (0.804, 0.113, 3.300),
    1: (0.969, 0.373, 4.949),
    2: (1.359, 0.624, 6.988),
}

# ============================================================
# TRIAL FILTER (per Goal.md, revised 2026-09-30)
# ============================================================
# Skip trials where the deep layer has no flow -- the project goal excludes
# the deep-flow-zero condition rather than fitting/training on it.
SKIP_DEEP_FLOW_ZERO = True


def print_summary():
    print("=== wachirawit.chai DCS parameters ===")
    print(f"mu_a={MU_A_MM:.4f} mm^-1  mu_s'={MU_SP_MM:.4f} mm^-1  "
          f"n={REFRACTIVE_INDEX_N}  k0={K0:.4f} mm^-1")
    print(f"geometry={DEFAULT_GEOMETRY}  shallow={SHALLOW_DEPTH_MM}mm  "
          f"deep={DEEP_DEPTH_MM}mm  voxel={VOXEL_SIZE_MM}mm")
    print(f"SDS={SDS_MM} mm  tau=0..{TAU_US[-1]:.0f}us ({N_TAU} pts)")
    print(f"default a={DEFAULT_A_MM2_S_PER_FR:.2e} mm^2/s per (ml/min/100g)  "
          f"D_B0={DEFAULT_DB0_MM2_S:.2e} mm^2/s")
    print(f"target c2/c1: {TARGET_C2_C1}")
    print(f"skip deep-flow==0 trials: {SKIP_DEEP_FLOW_ZERO}")


if __name__ == "__main__":
    print_summary()
