import sys, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
os.environ['NUMBA_NUM_THREADS'] = str(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print("NUMBA_NUM_THREADS set to: ", str(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
import jax
from jax import config
import numpy as np
from discodj import DiscoDJ
import jax.numpy as jnp
from jax._src.lib import xla_client
import gc
import MAS_library as MASL
import pickle as pk
import time
from numba import njit, prange

# =============================================================================
# TIMING UTILITIES
# =============================================================================
ENABLE_TIMING = False

class Timer:
    """Simple timer using explicit start/stop calls."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.records = {}
        self._start_times = {}
    
    def start(self, name):
        if self.enabled:
            self._start_times[name] = time.perf_counter()
    
    def stop(self, name):
        if not self.enabled:
            return
        if name not in self._start_times:
            return
        elapsed = time.perf_counter() - self._start_times[name]
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(elapsed)
        print(f"[TIMER] {name}: {elapsed:.4f}s")
        del self._start_times[name]
    
    def summary(self):
        if not self.enabled or not self.records:
            return
        print("\n" + "="*60)
        print("TIMING SUMMARY")
        print("="*60)
        for name, times in sorted(self.records.items(), key=lambda x: -sum(x[1])):
            total = sum(times)
            avg = total / len(times)
            print(f"{name:40s} | total: {total:8.3f}s | calls: {len(times):3d} | avg: {avg:.4f}s")
        print("="*60 + "\n")
    
    def reset(self):
        self.records = {}
        self._start_times = {}

timer = Timer(enabled=ENABLE_TIMING)

# =============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def _get_padded_mat_numba_core(Npart_pad, n_pad, grid_sbox, grid, fac):
    """
    Numba-parallelized extraction and block reduction.
    Avoids creating massive intermediate arrays.
    """
    fac3_inv = np.float32(1.0 / (fac * fac * fac))
    result = np.zeros((grid, grid, grid, grid_sbox, grid_sbox, grid_sbox), dtype=np.float32)
    
    # Parallelize over all grid cells
    for idx in prange(grid * grid * grid):
        gi = idx // (grid * grid)
        gj = (idx // grid) % grid
        gk = idx % grid
        
        # Starting position in padded array
        i0 = gi * grid_sbox
        j0 = gj * grid_sbox
        k0 = gk * grid_sbox
        
        for si in range(grid_sbox):
            i_base = i0 + si * fac
            for sj in range(grid_sbox):
                j_base = j0 + sj * fac
                for sk in range(grid_sbox):
                    k_base = k0 + sk * fac
                    
                    # Sum over the fac^3 block
                    total = np.float32(0.0)
                    for di in range(fac):
                        for dj in range(fac):
                            for dk in range(fac):
                                total += Npart_pad[i_base + di, j_base + dj, k_base + dk]
                    
                    result[gi, gj, gk, si, sj, sk] = total * fac3_inv
    
    return result


@njit(parallel=True, fastmath=True, cache=True)
def _velocity_field_numba(pos, vel, Npart_cic, BoxSize, grid_tot, norm_vel):
    """
    Compute velocity field with momentum weighting.
    Returns velocity field of shape (grid_tot, grid_tot, grid_tot, 3)
    """
    vel_field = np.zeros((grid_tot, grid_tot, grid_tot, 3), dtype=np.float32)
    mom_field = np.zeros((grid_tot, grid_tot, grid_tot, 3), dtype=np.float32)
    
    cell_size = BoxSize / grid_tot
    n_part = pos.shape[0]
    
    # CIC assignment for momentum
    for p in prange(n_part):
        # Get cell indices
        x = pos[p, 0] / cell_size
        y = pos[p, 1] / cell_size
        z = pos[p, 2] / cell_size
        
        # Integer and fractional parts
        i = int(x)
        j = int(y)
        k = int(z)
        
        dx = x - i
        dy = y - j
        dz = z - k
        
        # Wrap indices
        i0 = i % grid_tot
        j0 = j % grid_tot
        k0 = k % grid_tot
        i1 = (i + 1) % grid_tot
        j1 = (j + 1) % grid_tot
        k1 = (k + 1) % grid_tot
        
        # CIC weights
        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w010 = (1 - dx) * dy * (1 - dz)
        w011 = (1 - dx) * dy * dz
        w100 = dx * (1 - dy) * (1 - dz)
        w101 = dx * (1 - dy) * dz
        w110 = dx * dy * (1 - dz)
        w111 = dx * dy * dz
        
        for c in range(3):
            v = vel[p, c]
            mom_field[i0, j0, k0, c] += w000 * v
            mom_field[i0, j0, k1, c] += w001 * v
            mom_field[i0, j1, k0, c] += w010 * v
            mom_field[i0, j1, k1, c] += w011 * v
            mom_field[i1, j0, k0, c] += w100 * v
            mom_field[i1, j0, k1, c] += w101 * v
            mom_field[i1, j1, k0, c] += w110 * v
            mom_field[i1, j1, k1, c] += w111 * v
    
    # Divide by density to get velocity
    norm_vel_inv = np.float32(1.0 / norm_vel)
    for i in prange(grid_tot):
        for j in range(grid_tot):
            for k in range(grid_tot):
                n = Npart_cic[i, j, k]
                if n > 0:
                    for c in range(3):
                        vel_field[i, j, k, c] = mom_field[i, j, k, c] / n * norm_vel_inv
    
    return vel_field


def get_padded_mat_numba(Npart, n_pad, grid_sbox, grid):
    """Wrapper for numba-accelerated padded matrix computation."""
    timer.start("get_padded_mat:pad")
    Npart_pad = np.pad(Npart, n_pad, mode='wrap').astype(np.float32)
    timer.stop("get_padded_mat:pad")
    
    box_size = grid_sbox + 2 * n_pad
    fac = box_size // grid_sbox
    
    timer.start("get_padded_mat:numba_core")
    result = _get_padded_mat_numba_core(Npart_pad, n_pad, grid_sbox, grid, fac)
    timer.stop("get_padded_mat:numba_core")
    
    return result, None


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def mat_reshape_fast(mat, grid, grid_sbox):
    """Single reshape + transpose instead of multiple moveaxis"""
    grid_tot = grid * grid_sbox
    if mat.ndim == 3:
        return mat.reshape(grid, grid_sbox, grid, grid_sbox, grid, grid_sbox).transpose(0, 2, 4, 1, 3, 5)
    else:
        extra_dims = mat.shape[3:]
        return mat.reshape(grid, grid_sbox, grid, grid_sbox, grid, grid_sbox, *extra_dims).transpose(0, 2, 4, 1, 3, 5, *range(6, 6+len(extra_dims)))


def process_LH_sim_fast(pos_m_truth, vel_m_truth, get_env=False, get_vel=False, 
                        get_randsel=False, grid=64, grid_sbox=8, nrand_sel_box=32768):
    norm_delta = 10
    norm_vel = 100
    BoxSize = 1000.
    MAS_type = 'CIC'
    grid_tot = grid_sbox * grid

    rho_bar = len(pos_m_truth) / (BoxSize**3)
    vol_vox = (BoxSize / grid_tot)**3
    N_bar_vox = rho_bar * vol_vox

    # Density field
    timer.start("process:density_MAS")
    Npart = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
    pos_copy = np.array(pos_m_truth, dtype=np.float32, copy=True)
    MASL.MA(pos_copy, Npart, BoxSize, MAS_type, verbose=False)
    # Npart /= (N_bar_vox * norm_delta)
    timer.stop("process:density_MAS")

    timer.start("process:density_reshape")
    Npart_rs = mat_reshape_fast(Npart, grid, grid_sbox)
    timer.stop("process:density_reshape")
    
    fields_list = [Npart_rs[..., None]/(N_bar_vox * norm_delta), np.log1p(Npart_rs)[..., None]]
    
    if get_env or get_vel:
        timer.start("process:cic_for_vel_env")
        Npart_cic = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
        MASL.MA(pos_copy, Npart_cic, BoxSize, 'CIC', verbose=False)
        timer.stop("process:cic_for_vel_env")
    
    if get_env:
        # Use numba-accelerated version
        timer.start("process:env_pad1")
        Npart_pad1_rs, _ = get_padded_mat_numba(Npart, grid_sbox, grid_sbox, grid)
        timer.stop("process:env_pad1")
        
        timer.start("process:env_pad2")
        Npart_pad2_rs, _ = get_padded_mat_numba(Npart, 2*grid_sbox, grid_sbox, grid)
        timer.stop("process:env_pad2")
        fields_list.extend([Npart_pad1_rs[..., None]/(N_bar_vox * norm_delta), np.log1p(Npart_pad1_rs)[..., None], Npart_pad2_rs[..., None]/(N_bar_vox * norm_delta)])

    if get_vel:
        timer.start("process:velocity_field")
        vel_m_part = np.zeros((grid_tot, grid_tot, grid_tot, 3), dtype=np.float32)
        vel_copy = np.array(vel_m_truth, dtype=np.float32, copy=True)
        
        for jc in range(3):
            mom_jc = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
            MASL.MA(pos_copy, mom_jc, BoxSize, 'CIC', verbose=False, W=vel_copy[:, jc])
            
            vel_m_jc = np.divide(mom_jc, Npart_cic, out=np.zeros_like(mom_jc), 
                                  where=Npart_cic != 0)
            vel_m_part[..., jc] = vel_m_jc / norm_vel
        timer.stop("process:velocity_field")

        timer.start("process:velocity_reshape")
        vel_m_part_rs = mat_reshape_fast(vel_m_part, grid, grid_sbox)
        timer.stop("process:velocity_reshape")
        fields_list.append(vel_m_part_rs)

    timer.start("process:concatenate")
    dmo_fields_all_snap = np.concatenate(fields_list, axis=-1)
    dmo_fields_all_rs = dmo_fields_all_snap.reshape((grid**3, *dmo_fields_all_snap.shape[3:]))
    timer.stop("process:concatenate")

    # Random selection
    if (nrand_sel_box < grid**3) and get_randsel:
        timer.start("process:random_selection")
        rng = np.random.default_rng(0)
        Npart_sum = dmo_fields_all_rs[..., 0].sum(axis=(1, 2, 3))
        
        npart_min, npart_max = np.percentile(Npart_sum, [2.0, 98.0])
        Npart_clipped = np.clip(Npart_sum, npart_min, npart_max)
        
        hist, bins_edges = np.histogram(Npart_clipped, bins=16)
        bins_edges[0], bins_edges[-1] = 0.0, bins_edges[-1] * 100
        nsel_per_jb = nrand_sel_box // len(hist)

        indsel_all = []
        for jbd in range(len(bins_edges) - 1):
            indsel = np.where((Npart_sum >= bins_edges[jbd]) & 
                             (Npart_sum < bins_edges[jbd + 1]))[0]
            n_sel = min(len(indsel), nsel_per_jb)
            if n_sel > 0:
                indsel_all.append(rng.choice(indsel, n_sel, replace=False))

        indsel_all = np.concatenate(indsel_all)
        
        if len(indsel_all) < nrand_sel_box:
            remaining = np.setdiff1d(np.arange(grid**3), indsel_all)
            indsel_all = np.concatenate([
                indsel_all, 
                rng.choice(remaining, nrand_sel_box - len(indsel_all), replace=False)
            ])

        rand_sel = rng.permutation(indsel_all)
        Npart_sum_sel = Npart_sum[rand_sel]
        timer.stop("process:random_selection")
    else:
        rand_sel = np.arange(grid**3)
        Npart_sum = Npart_sum_sel = 0

    # any non-finite values set to zero:
    dmo_fields_all_rs[~np.isfinite(dmo_fields_all_rs)] = 0.0

    return dmo_fields_all_rs, rand_sel, Npart_sum, Npart_sum_sel, norm_delta, norm_vel


# =============================================================================
# CONFIGURATION
# =============================================================================
devices = jax.devices()
device = "gpu" if np.any([d.platform == "gpu" for d in devices]) else "cpu"
root = "/work/hdd/bdne/yzhang116/quijote/test_3gpc/"

dim = 3
precision = "single"
boxsize = 1000.0
res = 512
factor = 2
n_order = 2
a_ic = 1./128.

z_snaps = [2., 1., 0.5]
a_end_all = [1/(1+z_snaps[0]), 1/(1+z_snaps[1]), 1/(1+z_snaps[2])]
a_init_all = [a_ic, a_end_all[0], a_end_all[1]]
numsteps_all = [10, 5, 5]

stepper = "fastpm"
method = "pm"
res_pm = int(factor * res)
time_var = "D"
antialias = 0
grad_kernel_order = 4
laplace_kernel_order = 0
worder = 2
n_resample = 1
deconvolve = False
nlpt_order_ics = n_order
chunk_size = None

grid_sbox = 8
grid = 64
nrand_sel_box = 32768


# =============================================================================
# WARMUP NUMBA (compile before timing)
# =============================================================================
def warmup_numba():
    """Pre-compile numba functions to avoid JIT overhead in timing."""
    print("Warming up numba functions...")
    dummy = np.random.randn(64, 64, 64).astype(np.float32)
    dummy_pad = np.pad(dummy, 4, mode='wrap').astype(np.float32)
    _ = _get_padded_mat_numba_core(dummy_pad, 4, 8, 8, 3)
    print("Numba warmup complete.")


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================
def run_one_simulation(sim_id, savefull=False):
    path_ic = root + "ICs/"
    if savefull:
        root_out = root + "dmo/disco/"
    else:
        root_out = root + "dmo/disco/"

    os.makedirs(root_out, exist_ok=True)

    savefname_dmo_fields = root_out + 'dmo_fields_subvols_grid_%d_CV_%d_1gpc.npy' % (grid_sbox, sim_id)
    meta_fname = root_out + 'meta_dmo_fields_subvols_grid_%d_CV_%d_1gpc.pkl' % (grid_sbox, sim_id)

    if not os.path.exists(savefname_dmo_fields) or not os.path.exists(meta_fname):
        print("File missing for sim_id: ", sim_id)

        timer.start("load_cosmo_params")
        Om = 0.3175
        Ob = 0.049
        h = 0.6711
        ns = 0.9624
        sigma8 = 0.834
        Oc = Om - Ob
        timer.stop("load_cosmo_params")

        timer.start("load_ic_delta")
        # ic_delta = np.load(path_ic + "/IC_delta640.npy")
        ic_delta = np.load(path_ic + "IC_CV%d_1gpc.npy"%sim_id)

        ic_delta = jnp.array(ic_delta, dtype=jnp.float32)
        timer.stop("load_ic_delta")

        cosmo = dict(
            Omega_c=Oc,
            Omega_b=Ob,
            h=h,
            n_s=ns,
            sigma8=sigma8
        )

        timer.start("init_discodj")
        dj = DiscoDJ(dim=dim, res=res, device=device, precision=precision, boxsize=boxsize, cosmo=cosmo)
        dj = dj.with_timetables()
        Dplus_aic = np.interp(jnp.log10(a_ic), jnp.log10(dj.cosmo._timetables['a']), dj.cosmo._timetables['Dplus'])
        dj = dj.with_external_ics(delta=ic_delta / Dplus_aic)
        dj = dj.with_lpt(n_order=n_order, try_to_jit=True)
        timer.stop("init_discodj")

        for ja in range(len(a_end_all)):
            print("Running PM simulation for snap %d at a_end = %.3f" % (ja, a_end_all[ja]))
            
            timer.start(f"nbody_snap_{ja}")
            X_sim, P_sim, _ = dj.run_nbody(
                a_ini=a_init_all[ja], a_end=a_end_all[ja], n_steps=numsteps_all[ja], res_pm=res_pm,
                time_var=time_var, stepper=stepper, method=method,
                antialias=antialias, grad_kernel_order=grad_kernel_order,
                laplace_kernel_order=laplace_kernel_order,
                nlpt_order_ics=nlpt_order_ics, n_resample=n_resample,
                deconvolve=deconvolve, return_displacement=False,
                chunk_size=chunk_size)
            timer.stop(f"nbody_snap_{ja}")

            timer.start(f"post_nbody_convert_snap_{ja}")
            dj = dj.with_external_ics(pos=X_sim.reshape(-1, 3), vel=P_sim.reshape(-1, 3))
            P_sim = P_sim / a_end_all[ja] * 100.
            X_sim = np.asarray(X_sim).reshape(-1, 3)
            P_sim = np.asarray(P_sim).reshape(-1, 3)
            np.save(root_out+"pos_CV%d_z%.1f_1gpc.npy"%(sim_id, z_snaps[ja]), X_sim)
            np.save(root_out+"vel_CV%d_z%.1f_1gpc.npy"%(sim_id, z_snaps[ja]), P_sim)
            timer.stop(f"post_nbody_convert_snap_{ja}")

            if ja == 1:
                get_env, get_vel, get_randsel = True, True, False
            elif ja == 2:
                get_env, get_vel, get_randsel = True, True, True
            else:
                get_env, get_vel, get_randsel = False, False, False
            
            if savefull:
                nrand_sel = np.arange(grid**3)
                get_randsel = False

            timer.start(f"process_sim_snap_{ja}")
            dmo_fields_all_rs, rand_sel, Npart_sum, Npart_sum_sel, norm_delta, norm_vel = process_LH_sim_fast(
                X_sim, P_sim, get_env=get_env, get_vel=get_vel, get_randsel=get_randsel,
                grid_sbox=grid_sbox, grid=grid, nrand_sel_box=nrand_sel_box
            )
            timer.stop(f"process_sim_snap_{ja}")

            if ja == 0:
                dmo_fields_all_rs_all = dmo_fields_all_rs
            else:
                timer.start(f"concat_snap_{ja}")
                dmo_fields_all_rs_all = np.concatenate((dmo_fields_all_rs_all, dmo_fields_all_rs), axis=-1)
                timer.stop(f"concat_snap_{ja}")

            # if ja == 2:
            #     np.save(root_out+"/pos_LH%d_z05.npy"%sim_id,X_sim)
            #     np.save(root_out+"/vel_LH%d_z05.npy"%sim_id,P_sim)

            del X_sim, P_sim, dmo_fields_all_rs
            gc.collect()
            jax.clear_caches()

            print("Done snap %d" % (ja))

        print("Simulation %d done." % (sim_id))

        timer.start("save_outputs")
        np.save(savefname_dmo_fields, dmo_fields_all_rs_all.astype(np.float16)[rand_sel, ...])
        saved = {
            'cosmo':cosmo,
            'zsnaps': z_snaps,
            'Npart_sum_all': Npart_sum,
            'Npart_sum_sel': Npart_sum_sel,
            'norm_delta': norm_delta,
            'norm_vel': norm_vel,
            'grid_sbox': grid_sbox,
            'grid': grid,
            'rand_sel': rand_sel
        }
        pk.dump(saved, open(meta_fname, 'wb'))
        timer.stop("save_outputs")

        del dj, ic_delta
        gc.collect()
        jax.clear_caches()

        timer.summary()
        return
    else:
        return



# clear up jax caches
jax.clear_caches()
gc.collect()

sim_id = int(sys.argv[1])
try:
    savefull = bool(int(sys.argv[2]))
except:
    savefull = True
run_one_simulation(sim_id, savefull=savefull)