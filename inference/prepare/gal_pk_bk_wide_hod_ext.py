"""
Extended HOD galaxy catalog generator using Zheng07 + assembly bias + velocity bias.

HOD parameters (10 total):
  logMmin, sigma_logM, logM0, logM1, alpha   -- standard Zheng07
  abias_cen  -- central assembly bias,    N(0, 0.2) in [-1, 1]
  abias_sat  -- satellite assembly bias,  N(0, 0.2) in [-1, 1]
  eta_cen    -- central velocity bias,    U(0.0, 0.7)
  eta_sat    -- satellite velocity bias,  U(0.2, 2.0)
  eta_conc   -- satellite conc. bias,     U(0.2, 2.0)

Expected LH file columns (16 total):
  logMmin  sigma_logM  logM0  logM1  alpha
  abias_cen  abias_sat  eta_cen  eta_sat  eta_conc
  Om  Ob  h  ns  s8  LH_id_cosmo

Uses ltu-cmass phase-space / assembly-bias models via direct import (no omegaconf needed).
"""

import dill
import sys
import os
import numpy as np
import Pk_library as PKL
import MAS_library as MASL
from colossus.halo import mass_so
from colossus.cosmology import cosmology as colossus_cosmo
from astropy.cosmology import FlatLambdaCDM
from mpi4py import MPI

# ltu-cmass: import only the modules that don't require omegaconf
sys.path.insert(0, '/u/yzhang116/ltu-cmass')
from cmass.bias.tools.hod_models import Zheng07

from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import halo_mass_to_halo_radius, NFWProfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

save_pk_dir  = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/Pk_all_noise/'
save_gal_dir = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/galaxy_rsd_pos_noise/'
save_bk_dir  = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/Bk_all_noise/'

nthreads = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f"Rank {rank}: using {nthreads} threads", flush=True)
print("size:", size, flush=True)


# ---------------------------------------------------------------------------
# Halo catalog builder (inlined from ltu-cmass to avoid omegaconf dependency)
# ---------------------------------------------------------------------------

def _mass_to_concentration(mass, redshift, cosmo, mdef='vir'):
    model = NFWProfile(
        cosmology=cosmo,
        conc_mass_model='dutton_maccio14',
        mdef=mdef,
        redshift=redshift,
    )
    return model.conc_NFWmodel(prim_haloprop=mass)


def _build_halo_catalog(pos, vel, mass, redshift, BoxSize, cosmo,
                        conc=None, mdef='vir'):
    """Build a halotools UserSuppliedHaloCatalog.

    Parameters
    ----------
    pos  : (N,3) comoving Mpc/h
    vel  : (N,3) km/s  (physical)
    mass : (N,)  Msun/h  (linear, not log10)
    cosmo: astropy cosmology
    conc : (N,) halo NFW concentration; computed from mass-conc relation if None
    """
    mkey = f'halo_m{mdef}'
    rkey = f'halo_r{mdef}'

    radius = halo_mass_to_halo_radius(mass, cosmo, redshift, mdef)
    if conc is None:
        conc = _mass_to_concentration(mass, redshift, cosmo, mdef)

    kws = dict(
        halo_x=pos[:, 0], halo_y=pos[:, 1], halo_z=pos[:, 2],
        halo_vx=vel[:, 0], halo_vy=vel[:, 1], halo_vz=vel[:, 2],
        halo_nfw_conc=conc,
        halo_redshift=np.full(len(mass), redshift),
        halo_id=np.arange(len(mass)),
        halo_hostid=np.zeros(len(mass), dtype=int),
        halo_upid=np.full(len(mass), -1.0),
        halo_local_id=np.arange(len(mass), dtype='i8'),
        cosmology=cosmo,
        redshift=redshift,
        particle_mass=1,
        Lbox=BoxSize,
        mdef=mdef,
    )
    kws[mkey] = mass
    kws[rkey] = radius

    if mdef != 'vir':
        kws['halo_mvir'] = np.full(len(mass), np.nan)
        kws['halo_rvir'] = np.full(len(mass), np.nan)

    return UserSuppliedHaloCatalog(**kws)


def _build_HOD_model(cosmo, theta_hod, zf, mdef='vir'):
    """Instantiate and configure the extended Zheng07 HOD model."""
    model = Zheng07(mass_def=mdef, assem_bias=True, vel_assem_bias=True)
    model.set_parameters(dict(theta_hod))
    model.set_occupation()
    model.set_profiles(cosmology=cosmo, zf=zf)
    return model.get_model()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_halo_cats(isim, noise=0., z=0.5, boxsize=1000.,
                  LH_cosmo_val_file='/work/nvme/bdne/yzhang116/quijote_halos/quijote_params.txt'):

    LH_cosmo_val_all = np.loadtxt(LH_cosmo_val_file)
    Om, Ob, h, ns, s8 = LH_cosmo_val_all[isim]

    cosmo_astropy = FlatLambdaCDM(H0=100.0 * h, Om0=Om, Ob0=Ob)
    Ol = 1.0 - Om
    Hz = 100.0 * np.sqrt(Om * (1.0 + z)**3 + Ol)  # km/s/(Mpc/h)
    rsd_factor = (1.0 + z) / Hz

    mock_dir = '/work/hdd/bdne/spandey3/quijote_LH_discodj/quijote_generated_halo_cats'
    true_dir  = '/work/hdd/bdne/yzhang116/halo_catalogs_quijote_all'

    mock = np.load(mock_dir + f'/generated_halo_catalog_{isim}.npy')
    pos_mock = mock[:, :3].copy()
    if noise > 0:
        pos_mock += np.random.normal(0, noise, size=pos_mock.shape)
        pos_mock %= boxsize
    vel_mock  = mock[:, 4:7]
    mass_mock = mock[:, 3]
    conc_mock = mock[:, 7]

    true = np.loadtxt(true_dir + '/halo_LH_%d.dat' % isim)
    pos_truth  = true[:, 1:4]
    mass_truth = true[:, 0]
    vel_truth  = true[:, 4:7]
    rs_truth   = true[:, 7]

    # Concentration for truth halos: R_200c / r_s
    params_col = {'flat': True, 'H0': 100*h, 'Om0': Om, 'Ob0': Ob, 'sigma8': s8, 'ns': ns}
    colossus_cosmo.setCosmology('myCosmo', **params_col)
    Rhalo_truth = (1.0 + z) * mass_so.M_to_R(mass_truth, z, '200c')
    conc_truth  = Rhalo_truth / rs_truth

    halos_mock  = dict(pos=pos_mock,  vel=vel_mock,  mass=mass_mock, conc=conc_mock)
    halos_truth = dict(pos=pos_truth, vel=vel_truth, mass=mass_truth, conc=conc_truth)

    cosmo_attrs = dict(Om=Om, Ob=Ob, Ol=Ol, h=h, ns=ns, s8=s8,
                       Hz=Hz, rsd_factor=rsd_factor)

    return halos_mock, halos_truth, cosmo_astropy, cosmo_attrs


def get_theta_hod(isim, ihod,
                  LH_file_all='/work/hdd/bdne/yzhang116/quijote_galaxy/LH_points_HOD_cosmo_ext_20000.txt',
                  n_cosmo=2000):
    """
    Read one row from the Latin-hypercube file.

    File column order (16 columns):
      0-4   : logMmin  sigma_logM  logM0  logM1  alpha
      5-9   : abias_cen  abias_sat  eta_cen  eta_sat  eta_conc
      10-15 : Om  Ob  h  ns  s8  LH_id_cosmo
    """
    LH_points = np.loadtxt(LH_file_all)
    row = LH_points[isim + n_cosmo * ihod]

    logMmin, sigma_logM, logM0, logM1, alpha  = row[:5]
    abias_cen, abias_sat, eta_cen, eta_sat, eta_conc = row[5:10]
    Om, Ob, h, ns, s8, LH_id_cosmo           = row[10:]

    theta_hod = {
        'logMmin':    logMmin,
        'sigma_logM': sigma_logM,
        'logM0':      logM0,
        'logM1':      logM1,
        'alpha':      alpha,
        'mean_occupation_centrals_assembias_param1':   abias_cen,
        'mean_occupation_satellites_assembias_param1': abias_sat,
        'eta_vb_centrals':          eta_cen,
        'eta_vb_satellites':        eta_sat,
        'conc_gal_bias_satellites': eta_conc,
    }
    theta_cosmo = dict(Om=Om, Ob=Ob, h=h, ns=ns, sigma8=s8, LH_id_cosmo=LH_id_cosmo)

    return theta_hod, theta_cosmo


# ---------------------------------------------------------------------------
# Galaxy catalog generation
# ---------------------------------------------------------------------------

def get_gal_cats(halos, cosmo_astropy, theta_hod, z=0.5, boxsize=1000.,
                 seed=0, mdef='200c'):
    BoxSize = np.array([boxsize, boxsize, boxsize])

    catalog = _build_halo_catalog(
        halos['pos'], halos['vel'], halos['mass'],
        z, BoxSize, cosmo_astropy,
        conc=halos['conc'], mdef=mdef,
    )

    hod_factory = _build_HOD_model(cosmo_astropy, theta_hod, zf=z, mdef=mdef)
    hod_factory.populate_mock(
        catalog, seed=seed,
        halo_mass_column_key=f'halo_m{mdef}',
    )

    gal = hod_factory.mock.galaxy_table
    pos      = np.array([gal['x'],  gal['y'],  gal['z']]).T
    vel      = np.array([gal['vx'], gal['vy'], gal['vz']]).T
    gal_type = (np.array(gal['gal_type']) == 'satellites').astype(int)  # 0=cen, 1=sat

    n_total = len(pos)
    n_sat   = int(gal_type.sum())
    galsum  = dict(
        total=n_total,
        number_density=n_total / boxsize**3,
        centrals=n_total - n_sat,
        satellites=n_sat,
        fsat=n_sat / n_total,
    )
    return pos, vel, gal_type, galsum


def get_gal_pos(pos, vel, rsd_factor, pos_type='rsd', boxsize=1000.,
                los=(1, 0, 0), noise=0.):
    los = np.asarray(los, dtype=float)
    if pos_type == 'rsd':
        out = pos + vel * rsd_factor * los
    else:
        out = pos.copy()

    out = out.astype(np.float32) % boxsize

    if noise > 0:
        out += np.random.normal(0, noise, size=out.shape).astype(np.float32)
        out %= boxsize

    return out


# ---------------------------------------------------------------------------
# Power spectrum and bispectrum
# ---------------------------------------------------------------------------

def get_gal_Pk(gal_pos, boxsize=1000., grid=512, MAS='NGP', compensated=False):
    mesh = np.zeros((grid, grid, grid), dtype=np.float32)
    MASL.MA(gal_pos, mesh, boxsize, MAS)
    mesh /= np.mean(mesh, dtype=np.float32)
    mesh -= 1.0
    MAS_arg = MAS if compensated else None
    Pk = PKL.Pk(mesh, boxsize, axis=0, MAS=MAS_arg, threads=nthreads)
    return Pk, mesh


def get_gal_Bk(k1, k2, theta, pos, boxsize=1000., grid=256, MAS='NGP', compensated=False):
    mesh = np.zeros((grid, grid, grid), dtype=np.float32)
    MASL.MA(pos, mesh, boxsize, MAS)
    mesh /= np.mean(mesh, dtype=np.float32)
    mesh -= 1.0
    MAS_arg = MAS if compensated else None
    Bk = PKL.Bk(mesh, boxsize, k1, k2, theta, MAS_arg, nthreads)
    return Bk


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def get_Pk_Bk_all_hods(isim, noise=0., nhod_LH_samp=10, kmax=1.0):
    saved_j   = {'isim': isim}
    saved_gal = {'isim': isim}
    saved_k   = {'isim': isim}

    halos_mock, halos_truth, cosmo_astropy, cosmo_attrs = get_halo_cats(isim, noise=0.)
    rsd_factor = cosmo_attrs['rsd_factor']

    k1_array = [0.08, 0.16, 0.32]
    theta_bk = np.linspace(0.1, np.pi - 0.1, 8)

    for ihod in range(nhod_LH_samp):
        theta_hod, theta_cosmo = get_theta_hod(isim, ihod=ihod)

        for key in ('theta_hod', 'theta_cosmo'):
            val = theta_hod if key == 'theta_hod' else theta_cosmo
            for d in (saved_j, saved_gal, saved_k):
                d[f'{key}_{ihod}'] = val

        pos_mock,  vel_mock,  _, galsum_mock  = get_gal_cats(halos_mock,  cosmo_astropy, theta_hod)
        pos_truth, vel_truth, _, galsum_truth = get_gal_cats(halos_truth, cosmo_astropy, theta_hod)

        for key in ('galsum_mock', 'galsum_truth'):
            val = galsum_mock if 'mock' in key else galsum_truth
            for d in (saved_j, saved_gal, saved_k):
                d[f'{key}_{ihod}'] = val

        rsd_pos_mock  = get_gal_pos(pos_mock,  vel_mock,  rsd_factor, noise=noise)
        rsd_pos_truth = get_gal_pos(pos_truth, vel_truth, rsd_factor, noise=noise)

        # saved_gal[f'pos_rsd_mock_{ihod}']  = rsd_pos_mock
        # saved_gal[f'pos_rsd_truth_{ihod}'] = rsd_pos_truth

        # Power spectrum
        Pk_mock,  _ = get_gal_Pk(rsd_pos_mock)
        Pk_truth, _ = get_gal_Pk(rsd_pos_truth)
        sel = np.where((Pk_mock.k3D >= 0.01) & (Pk_mock.k3D <= kmax))[0]
        saved_k[f'rsd_Pk_mock_{ihod}']  = Pk_mock.Pk[sel, :]
        saved_k[f'rsd_Pk_truth_{ihod}'] = Pk_truth.Pk[sel, :]
        saved_k[f'k_Pk_{ihod}']         = Pk_mock.k3D[sel]

        # Bispectrum
        for k1 in k1_array:
            Bk_mock  = get_gal_Bk(k1, k1, theta_bk, rsd_pos_mock)
            Bk_truth = get_gal_Bk(k1, k1, theta_bk, rsd_pos_truth)
            tag = f'0p{int(k1*100):02d}'
            saved_j[f'rsd_Bk_mock_{tag}_{ihod}']  = Bk_mock.B
            saved_j[f'rsd_Bk_truth_{tag}_{ihod}'] = Bk_truth.B
            saved_j[f'rsd_Qk_mock_{tag}_{ihod}']  = Bk_mock.Q
            saved_j[f'rsd_Qk_truth_{tag}_{ihod}'] = Bk_truth.Q

    saved_j['theta_Bk'] = theta_bk
    saved_j['k_array']  = k1_array

    dill.dump(saved_k,   open(save_pk_dir  + f'Pk_NGP_galnoise_{noise:.2f}_LH_{isim}.dill',  'wb'))
    # dill.dump(saved_gal, open(save_gal_dir + f'Galaxy_pos_galnoise_{noise:.2f}_LH_{isim}.dill', 'wb'))
    dill.dump(saved_j,   open(save_bk_dir  + f'Bk_NGP_galnoise_{noise:.2f}_LH_{isim}.dill',  'wb'))


if __name__ == '__main__':
    start = int(sys.argv[1])
    end   = int(sys.argv[2])
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello from rank {rank}/{size}", flush=True)

    for simid in range(start + rank, end, size):
        print(f"Rank {rank} processing simulation {simid}", flush=True)
        get_Pk_Bk_all_hods(simid, noise=noise, nhod_LH_samp=10, kmax=1.0)
