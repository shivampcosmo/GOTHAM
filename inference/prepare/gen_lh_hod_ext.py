"""
Generate a Latin Hypercube sampling file for the 10-parameter extended HOD model.

Output file columns (16 total):
  0  logMmin       -- N(12.35, 0.13)  truncated to [12.0, 14.0]
  1  sigma_logM    -- U(0.1,  0.6)
  2  logM0         -- U(13.0, 15.0)
  3  logM1         -- U(13.0, 15.0)
  4  alpha         -- U(0.0,  1.5)
  5  abias_cen     -- N(0, 0.2)       truncated to [-1, 1]
  6  abias_sat     -- N(0, 0.2)       truncated to [-1, 1]
  7  eta_cen       -- U(0.0,  0.7)
  8  eta_sat       -- U(0.2,  2.0)
  9  eta_conc      -- U(0.2,  2.0)
  10 Om
  11 Ob
  12 h
  13 ns
  14 s8
  15 LH_id_cosmo

Row index: isim + N_COSMO * ihod  (matches get_theta_hod indexing)
"""

import numpy as np
from scipy.stats import truncnorm
from scipy.stats.qmc import LatinHypercube

# ── Configuration ──────────────────────────────────────────────────────────
N_COSMO          = 2000
N_HOD_PER_COSMO  = 10
BASE_SEED        = 42

COSMO_FILE = '/work/nvme/bdne/yzhang116/quijote_halos/quijote_params.txt'
OUT_FILE   = '/work/hdd/bdne/yzhang116/quijote_galaxy/LH_points_HOD_cosmo_ext_wideflat_20000.txt'

# ── Parameter priors (order matches output columns 0-9) ───────────────────
#
# Each entry: ('type', *args)
#   ('truncnorm', loc, scale, low, high)
#   ('uniform',   low, high)
#
PRIORS = [
    ('truncnorm', 12.35, 0.13,  12.0,  14.0),   # logMmin
    ('uniform',    0.1,   0.6),                   # sigma_logM
    ('uniform',   13.0,  15.0),                   # logM0
    ('uniform',   13.0,  15.0),                   # logM1
    ('uniform',    0.0,   1.5),                   # alpha
    ('truncnorm',  0.0,   0.2,  -1.0,   1.0),    # abias_cen
    ('truncnorm',  0.0,   0.2,  -1.0,   1.0),    # abias_sat
    ('uniform',    0.0,   0.7),                   # eta_cen
    ('uniform',    0.2,   2.0),                   # eta_sat
    ('uniform',    0.2,   2.0),                   # eta_conc
]

N_HOD_PARAMS = len(PRIORS)


def _apply_prior(u_col, prior):
    """Transform one column of uniform [0,1] samples via the prior's inverse CDF."""
    kind = prior[0]
    if kind == 'uniform':
        lo, hi = prior[1], prior[2]
        return lo + u_col * (hi - lo)
    elif kind == 'truncnorm':
        loc, scale, lo, hi = prior[1], prior[2], prior[3], prior[4]
        a, b = (lo - loc) / scale, (hi - loc) / scale
        return truncnorm.ppf(u_col, a, b, loc=loc, scale=scale)
    else:
        raise ValueError(f'Unknown prior type: {kind}')


def sample_hod_lhc(n, seed):
    """Return (n, N_HOD_PARAMS) array of HOD parameter samples via LHC."""
    sampler = LatinHypercube(d=N_HOD_PARAMS, seed=seed)
    u = sampler.random(n)          # (n, N_HOD_PARAMS) uniform in [0, 1]
    params = np.column_stack([
        _apply_prior(u[:, i], PRIORS[i]) for i in range(N_HOD_PARAMS)
    ])
    return params


def main():
    cosmo_all = np.loadtxt(COSMO_FILE)          # (>=N_COSMO, 5): Om Ob h ns s8
    cosmo_all = cosmo_all[:N_COSMO]
    lh_ids    = np.arange(N_COSMO, dtype=float).reshape(-1, 1)
    cosmo_block = np.hstack([cosmo_all, lh_ids])  # (N_COSMO, 6)

    blocks = []
    for ihod in range(N_HOD_PER_COSMO):
        hod_params = sample_hod_lhc(N_COSMO, seed=BASE_SEED + ihod)
        blocks.append(np.hstack([hod_params, cosmo_block]))

    out = np.vstack(blocks)          # (N_COSMO * N_HOD_PER_COSMO, 16)

    header = (
        'logMmin sigma_logM logM0 logM1 alpha '
        'abias_cen abias_sat eta_cen eta_sat eta_conc '
        'Om Ob h ns s8 LH_id_cosmo'
    )
    np.savetxt(OUT_FILE, out, fmt='%.6f', header=header)
    print(f'Saved {out.shape[0]} rows x {out.shape[1]} cols -> {OUT_FILE}')


if __name__ == '__main__':
    main()
