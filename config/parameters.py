"""
Physical and simulation parameters of Model A (DCS two-layer flow phantom).
Single source of truth -- every module in src/ imports its constants from here.

Sources:
  - Nakabayashi et al. (2023), Biomed. Opt. Express 14(10) 5358-5375
  - Wu et al. (2021), Neurophotonics 8(1) 015001
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
# PHANTOM GEOMETRY (Nakabayashi Sec. 2.1)
# ============================================================
# The fluid volume at the paper's stated size: 100 x 50 mm footprint, shallow 5.5 mm +
# deep 35 mm. Acrylic wall thickness is not given, so the fluid fills the full footprint;
# the 0.05 mm Mylar sheet is not voxelised (it would become a 10x-too-thick 0.5 mm voxel).
# The phantom is a closed box, so every face is Fresnel ('r': fluid n=1.37 vs outside 1.0).
# Probe placement is not given: the source-to-3cm-detector span is centred on the 100 mm
# length (source at 35 mm, detectors at 45/55/65 mm), y at the centre.
VOXEL_SIZE_MM = 0.5
PHANTOM_X_MM = 100.0
PHANTOM_Y_MM = 50.0
SHALLOW_DEPTH_MM = 5.5
DEEP_DEPTH_MM = 35.0
FLUID_Z_MM = SHALLOW_DEPTH_MM + DEEP_DEPTH_MM      # 40.5
SOURCE_X_MM = (PHANTOM_X_MM - 30.0) / 2.0          # 35.0
MCX_BC = "rrrrrr"          # faces -x,-y,-z,+x,+y,+z (mcxcl README "--bc")

# ============================================================
# SOURCE / DETECTOR
# ============================================================
SDS_MM = np.array([10.0, 20.0, 30.0])      # 1, 2, 3 cm
SDS_LABELS = ["1cm", "2cm", "3cm"]
DET_RADIUS_MM = 1.0
NPHOTON = 1e8              # enough: 1e7..1e8 give the same R^2 (photon sweep, Obsidian)
PHOTONS_PER_SDS = 5000     # detected photons used per SDS when building g1

# ============================================================
# DCS DELAY TIME AXIS
# ============================================================
N_TAU = 501
TAU_S = np.arange(N_TAU, dtype=float) * 1e-6     # 0, 1, ..., 500 us

# ============================================================
# DYNAMICS: flow -> alpha*D_B
# ============================================================
# alpha*D_B = a * max(FR, 0) + D_B0
#   a    : one scale shared by both layers (one physical fluid/bead matrix)
#   D_B0 : Brownian baseline so g2 still decays at FR = 0
# (a, D_B0) are fitted by Nelder-Mead (src/calibrate.py); the values below are only
# the starting point of that search.
START_A_MM2_S_PER_FR = 1e-6
START_DB0_MM2_S = 4.5e-7
CAL_BOUNDS_LOG10 = [(-9.0, -3.0), (-9.0, -5.0)]   # log10 a, log10 D_B0
CAL_TAU_STRIDE = 5          # calibrate on every 5th tau point (101 of 501) for speed


def flow_to_alpha_db(flow, a, db0):
    """alpha*D_B (mm^2/s) = a * max(flow, 0) + D_B0. Negative measured flow (scale noise
    at FR = 0) is clamped to zero rather than allowed to subtract from the baseline."""
    return a * np.maximum(flow, 0.0) + db0


# ============================================================
# SEMI-INFINITE HOMOGENEOUS MODEL (BFI fitting, paper Eq. 3)
# ============================================================
Z0_MM = 1.0 / MU_SP_MM
# zb = 1.76/mu_s' for n=1.37, as given by Wu et al. 2021 (Nakabayashi's ref [37]).
ZB_MM = 1.76 / MU_SP_MM
BFI_MM2S_TO_FILE = 1e7      # fitter returns mm^2/s; metadata BFI is in 1e-9 cm^2/s
