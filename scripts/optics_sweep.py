"""Diagnostic: does lowering the MC medium's mu_s'/mu_a (fluid volume fraction in
the glass-bead packing) raise synthetic c2/c1 toward the physical same-fitter value?
The BFI fitter keeps the paper's nominal fluid optics, as the paper's analysis would."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
import numpy as np
from config import parameters as cfg
import data_io, bfi_fit, mcx_engine
from dcs_physics import calculate_g2

NOMINAL_MUS, NOMINAL_MUA = cfg.MU_S_MM, cfg.MU_A_MM
flow = data_io.load_flow_table()
g2p = data_io.load_physical_g2()
beta = data_io.extract_beta(g2p)
tau = data_io.TAU_ARRAY
rng = np.random.default_rng(1)
idx = rng.choice(flow["n_trials"], 250, replace=False)
fs, fd = flow["fr_meas_s"][idx], flow["fr_meas_d"][idx]

for frac in (1.0, 0.6, 0.4):
    cfg.MU_S_MM, cfg.MU_A_MM = NOMINAL_MUS * frac, NOMINAL_MUA * frac
    paths = mcx_engine.run_simulation(nphoton=5e7)
    mua = cfg.MU_A_MM
    cfg.MU_S_MM, cfg.MU_A_MM = NOMINAL_MUS, NOMINAL_MUA   # fitter uses nominal optics
    det = {s: data_io.downsample_detector(paths, s + 1, 5000, rng) for s in range(3)}
    sim = np.zeros((3, len(tau), len(idx)))
    for j in range(len(idx)):
        a_s = cfg.flow_to_alpha_db(fs[j], 1.3e-7, 4.5e-7)
        a_d = cfg.flow_to_alpha_db(fd[j], 1.3e-7, 4.5e-7)
        for s in range(3):
            sim[s, :, j] = calculate_g2(*det[s], a_s, a_d, tau, beta[s], mua_s=mua, mua_d=mua)
    res = bfi_fit.regress_c2c1(bfi_fit.fit_bfi_all(sim, beta, tau), fs, fd)
    counts = [len(det_full) for det_full in (paths["det_1_Ls"], paths["det_2_Ls"], paths["det_3_Ls"])]
    print(f"RESULT optics x{frac}: photons={counts}  c2/c1 = "
          + " / ".join(f"{r[4]:.3f}" for r in res), flush=True)
