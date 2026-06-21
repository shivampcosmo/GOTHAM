"""
Multi-GPU DISCO-DJ PM simulation — 3 snapshots (z=2, 1, 0.5).
Uses multiple GPUs via discodj_dist for both the distributed nLPT initial state
and the distributed PM (DKD-Pi) evolution.
Paths, cosmology, and post-processing match NN/prepare/pm.py.
"""
import sys, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['NUMBA_NUM_THREADS'] = str(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print("NUMBA_NUM_THREADS:", os.environ.get("NUMBA_NUM_THREADS", 1), flush=True)
# The AWS OFI plugin (libnccl-net.so) can require libfabric symbols that are
# absent here, so unset the plugin pointer and let NCCL choose its built-in
# P2P/SHM/socket transports.
os.environ.pop('NCCL_NET_PLUGIN', None)

import gc
import time
import threading
import subprocess
import socket
import struct
import errno
import psutil
from functools import partial
import numpy as np
import jax


def _get_env_int(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def _get_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_slurm_coordinator_address() -> str:
    """Return host:port for JAX's coordinator from env or SLURM_NODELIST."""
    explicit = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if explicit:
        return explicit

    host = os.environ.get("JAX_COORDINATOR_HOST")
    if not host:
        nodelist = os.environ.get("SLURM_NODELIST")
        if not nodelist:
            raise RuntimeError(
                "Multi-process JAX requested but JAX_COORDINATOR_ADDRESS is not set "
                "and SLURM_NODELIST is unavailable."
            )
        host = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
        ).splitlines()[0]

    port = os.environ.get("JAX_COORDINATOR_PORT", "12355")
    return host if ":" in host else f"{host}:{port}"


def _parse_int_list(value: str) -> list[int] | None:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        return None


def _get_jax_local_device_ids() -> list[int] | None:
    """Use all GPUs assigned to this Slurm task for one-process-per-node JAX."""
    explicit = os.environ.get("JAX_LOCAL_DEVICE_IDS")
    if explicit:
        parsed = _parse_int_list(explicit)
        if parsed is None:
            raise ValueError(f"JAX_LOCAL_DEVICE_IDS must be comma-separated ints, got {explicit!r}")
        return parsed

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible and cuda_visible not in {"NoDevFiles", "none", "void"}:
        tokens = [item.strip() for item in cuda_visible.split(",") if item.strip()]
        if tokens:
            # CUDA_VISIBLE_DEVICES remaps the task's assigned GPUs into a local
            # 0..N-1 namespace, even when the original identifiers are UUIDs.
            return list(range(len(tokens)))

    slurm_step_gpus = os.environ.get("SLURM_STEP_GPUS")
    if slurm_step_gpus:
        parsed = _parse_int_list(slurm_step_gpus)
        if parsed is not None:
            return parsed

    return None


def _initialize_jax_distributed() -> bool:
    """Initialize JAX distributed when launched as one Slurm task per node."""
    num_processes = _get_env_int(
        "JAX_NUM_PROCESSES",
        "JAX_PROCESS_COUNT",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        default=1,
    )
    process_id = _get_env_int(
        "JAX_PROCESS_ID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        default=0,
    )
    if num_processes <= 1:
        return False

    coordinator_address = _get_slurm_coordinator_address()
    local_device_ids = _get_jax_local_device_ids()
    print(
        "Initializing JAX distributed: "
        f"process_id={process_id}, num_processes={num_processes}, "
        f"coordinator={coordinator_address}, "
        f"local_device_ids={local_device_ids}",
        flush=True,
    )
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
    )
    return True


_JAX_DISTRIBUTED = _initialize_jax_distributed()
import jax.numpy as jnp
from jax import lax
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import MAS_library as MASL
import pickle as pk
from numba import njit, prange
MPI = None

# discodj_dist lives next to the user's discodj install
_DISCODJ_SRC = '/mnt/ceph/users/spandey/quijote_v2_gotham/DISCO-DJ/src'
if _DISCODJ_SRC not in sys.path:
    sys.path.insert(0, _DISCODJ_SRC)
import jaxdecomp


def _configure_jaxdecomp_from_env() -> None:
    """Apply optional jaxDecomp backend tuning before jaxdecomp.init()."""
    transpose_backend = os.environ.get("DISCO_JAXDECOMP_TRANSPOSE_BACKEND", "").strip().upper()
    if transpose_backend:
        transpose_map = {
            "NCCL": jaxdecomp.TRANSPOSE_COMM_NCCL,
            "NCCL_PL": jaxdecomp.TRANSPOSE_COMM_NCCL_PL,
            "MPI_A2A": jaxdecomp.TRANSPOSE_COMM_MPI_A2A,
            "MPI_P2P": jaxdecomp.TRANSPOSE_COMM_MPI_P2P,
            "MPI_P2P_PL": jaxdecomp.TRANSPOSE_COMM_MPI_P2P_PL,
            "NVSHMEM": jaxdecomp.TRANSPOSE_COMM_NVSHMEM,
            "NVSHMEM_PL": jaxdecomp.TRANSPOSE_COMM_NVSHMEM_PL,
        }
        try:
            jaxdecomp.config.update("transpose_comm_backend", transpose_map[transpose_backend])
        except KeyError as exc:
            raise ValueError(
                "DISCO_JAXDECOMP_TRANSPOSE_BACKEND must be one of "
                f"{sorted(transpose_map)}, got {transpose_backend!r}"
            ) from exc

    halo_backend = os.environ.get("DISCO_JAXDECOMP_HALO_BACKEND", "").strip().upper()
    if halo_backend:
        halo_map = {
            "NCCL": jaxdecomp.HALO_COMM_NCCL,
            "MPI": jaxdecomp.HALO_COMM_MPI,
            "MPI_BLOCKING": jaxdecomp.HALO_COMM_MPI_BLOCKING,
            "NVSHMEM": jaxdecomp.HALO_COMM_NVSHMEM,
            "NVSHMEM_BLOCKING": jaxdecomp.HALO_COMM_NVSHMEM_BLOCKING,
        }
        try:
            jaxdecomp.config.update("halo_comm_backend", halo_map[halo_backend])
        except KeyError as exc:
            raise ValueError(
                "DISCO_JAXDECOMP_HALO_BACKEND must be one of "
                f"{sorted(halo_map)}, got {halo_backend!r}"
            ) from exc

    axis_contiguous = os.environ.get("DISCO_JAXDECOMP_TRANSPOSE_AXIS_CONTIGUOUS")
    if axis_contiguous not in (None, ""):
        jaxdecomp.config.update(
            "transpose_axis_contiguous",
            axis_contiguous.strip().lower() not in {"0", "false", "no", "off"},
        )


_configure_jaxdecomp_from_env()
if _JAX_DISTRIBUTED:
    jaxdecomp.init()
    print(
        f"jaxdecomp initialized on process {jax.process_index()} / {jax.process_count()}",
        flush=True,
    )
from discodj_dist import DiscoDJ
from discodj_dist.nbody.steppers.dkd_pi_integrator import DKDPiIntegrator
from discodj_dist.nbody.acc_distributed import kick_PM_distributed
from discodj_dist.core.distributed_pm import get_local_q_grid, gradient_kernel_dist
from discodj_dist.lpt.nlpt_distributed import (
    build_k_vecs_dist,
    compute_2lpt_initial_state_distributed,
)
import discodj_dist.lpt.nlpt_distributed as _nlpt_dist

all_gather = partial(multihost_utils.process_allgather, tiled=True)


# =============================================================================
# MEMORY MONITOR
# =============================================================================
class MemoryMonitor:
    """Tracks peak CPU RSS in a background thread; reads JAX GPU peak on demand."""
    def __init__(self, interval: float = 1.0):
        self._proc     = psutil.Process(os.getpid())
        self._interval = interval
        self._peak_cpu = 0
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._peak_cpu = self._proc.memory_info().rss
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _poll(self):
        while not self._stop.wait(self._interval):
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak_cpu:
                    self._peak_cpu = rss
            except Exception:
                pass

    @property
    def peak_cpu_gb(self) -> float:
        return self._peak_cpu / 1e9

    def gpu_peak_gb(self) -> dict:
        """Per-device peak GPU memory in GB from JAX memory_stats."""
        out = {}
        for dev in jax.local_devices():
            stats = dev.memory_stats()
            if stats:
                peak = stats.get('peak_bytes_in_use', stats.get('bytes_in_use', 0))
                out[str(dev)] = peak / 1e9
        return out

    def log_summary(self):
        print(f"  [mem] Peak CPU RAM : {self.peak_cpu_gb:.2f} GB", flush=True)
        for dev, gb in self.gpu_peak_gb().items():
            print(f"  [mem] Peak GPU ({dev}): {gb:.2f} GB", flush=True)


# =============================================================================
# CONFIGURATION  (paths / params mirror pm.py)
# =============================================================================
root     = "/mnt/ceph/users/spandey/quijote_v2_gotham/IC_3gpc_test/"
path_ic  = root + "ICs/"
root_out = root + "dmo/disco/"

dim       = 3
precision = "single"
boxsize   = float(os.environ.get("DISCO_BOXSIZE", "3000.0"))
res       = int(os.environ.get("DISCO_RES", "1536"))
factor    = int(os.environ.get("DISCO_PM_FACTOR", "2"))
n_order   = int(os.environ.get("DISCO_LPT_ORDER", "2"))
a_ic      = 1. / 128.
if n_order < 1:
    raise ValueError(f"DISCO_LPT_ORDER must be >= 1, got {n_order}.")

# Three staged snapshots, matching pm.py exactly
z_snaps      = [2., 1., 0.5]
a_end_all    = [1. / (1. + z) for z in z_snaps]
a_init_all   = [a_ic, a_end_all[0], a_end_all[1]]
numsteps_all = [10, 5, 5]

stepper              = "fastpm"
method               = "pm"
res_pm               = factor * res 
time_var             = "D"
antialias            = 0
grad_kernel_order    = int(os.environ.get("DISCO_PM_GRAD_KERNEL_ORDER", "4"))
lpt_grad_kernel_order = int(os.environ.get("DISCO_LPT_GRAD_KERNEL_ORDER", "0"))
laplace_kernel_order = 0
worder               = 2
n_resample           = 1
deconvolve           = False
ic_fft_backend       = os.environ.get("DISCO_IC_FFT_BACKEND", "JAX").strip()
lpt_fft_backend      = os.environ.get("DISCO_LPT_FFT_BACKEND", "JAX").strip()
lpt_mu2_fft_backend = os.environ.get("DISCO_LPT_MU2_FFT_BACKEND", "").strip()
pm_fft_backend       = os.environ.get("DISCO_PM_FFT_BACKEND", "JAX").strip()
for _backend_name, _backend_value in (
    ("DISCO_IC_FFT_BACKEND", ic_fft_backend),
    ("DISCO_LPT_FFT_BACKEND", lpt_fft_backend),
):
    if _backend_value.lower() not in {"jax", "cudecomp"}:
        raise ValueError(f"{_backend_name} must be 'JAX' or 'cudecomp', got {_backend_value!r}")
if pm_fft_backend.lower() not in {"jax", "cudecomp", "jax_rfft", "rfft", "slab_rfft"}:
    raise ValueError(
        "DISCO_PM_FFT_BACKEND must be 'JAX', 'cudecomp', or 'JAX_RFFT', "
        f"got {pm_fft_backend!r}"
    )
ic_fft_backend = "cudecomp" if ic_fft_backend.lower() == "cudecomp" else "JAX"
lpt_fft_backend = "cudecomp" if lpt_fft_backend.lower() == "cudecomp" else "JAX"
allow_experimental_cudecomp_lpt = _get_env_bool("DISCO_ALLOW_EXPERIMENTAL_CUDECOMP_LPT_FFT", False)
if (ic_fft_backend == "cudecomp" or lpt_fft_backend == "cudecomp") and not allow_experimental_cudecomp_lpt:
    raise RuntimeError(
        "cuDecomp is currently not a validated IC/LPT FFT backend for this "
        "production path. Diagnostics found that mixed JAX-fphi/cuDecomp-LPT "
        "corrupts the real-space displacement layout, and cuDecomp IC+LPT "
        "under-normalizes the initial momentum. Use DISCO_IC_FFT_BACKEND=JAX "
        "and DISCO_LPT_FFT_BACKEND=JAX. Set "
        "DISCO_ALLOW_EXPERIMENTAL_CUDECOMP_LPT_FFT=1 only for explicit backend "
        "diagnostics, not production runs."
    )
if pm_fft_backend.lower() == "cudecomp":
    pm_fft_backend = "cudecomp"
elif pm_fft_backend.lower() in {"jax_rfft", "rfft", "slab_rfft"}:
    pm_fft_backend = "JAX_RFFT"
else:
    pm_fft_backend = "JAX"
if not lpt_mu2_fft_backend:
    lpt_mu2_fft_backend = "JAX" if lpt_fft_backend == "cudecomp" else lpt_fft_backend
elif lpt_mu2_fft_backend.lower() not in {"jax", "cudecomp"}:
    raise ValueError(
        "DISCO_LPT_MU2_FFT_BACKEND must be 'JAX' or 'cudecomp', "
        f"got {lpt_mu2_fft_backend!r}"
    )
lpt_mu2_fft_backend = "cudecomp" if lpt_mu2_fft_backend.lower() == "cudecomp" else "JAX"
# For factor=2, PM cell size = boxsize/res_pm ≈ 0.977 Mpc/h
# (same as 512^3/1000/1024 case).  On 4 GPUs, factor=2 is generally
# too memory-heavy for the distributed PM FFT at this resolution; use the
# multi-node Slurm script when DISCO_PM_FACTOR=2.
halo_size = int(os.environ.get("DISCO_HALO_SIZE", "80"))
halo_safety_cells = float(os.environ.get("DISCO_HALO_SAFETY_CELLS", "2"))
validate_halo = _get_env_bool("DISCO_VALIDATE_HALO", True)
output_tag = os.environ.get("DISCO_OUTPUT_TAG", "3gpc_multigpu").strip() or "3gpc_multigpu"
diagnostic_mode = os.environ.get("DISCO_DIAGNOSTIC_MODE", "").strip().lower()
valid_diagnostic_modes = {
    "",
    "initial_state_stats",
    "sampled_density",
    "sampled_density_env",
    "sampled_density_velocity",
    "state_stats",
}
if diagnostic_mode not in valid_diagnostic_modes:
    raise ValueError(
        f"DISCO_DIAGNOSTIC_MODE must be one of {sorted(valid_diagnostic_modes)}, "
        f"got {diagnostic_mode!r}"
    )
diagnostic_sampled = diagnostic_mode in {
    "sampled_density",
    "sampled_density_env",
    "sampled_density_velocity",
}
diagnostic_state_stats = diagnostic_mode == "state_stats"
diagnostic_initial_state_stats = diagnostic_mode == "initial_state_stats"
diagnostic_step_stats = _get_env_bool("DISCO_DIAGNOSTIC_STEP_STATS", False)
diagnostic_max_pm_steps_env = os.environ.get("DISCO_DIAGNOSTIC_MAX_PM_STEPS", "").strip()
diagnostic_max_pm_steps = None
if diagnostic_max_pm_steps_env:
    diagnostic_max_pm_steps = int(diagnostic_max_pm_steps_env)
    if diagnostic_max_pm_steps < 0:
        raise ValueError(
            f"DISCO_DIAGNOSTIC_MAX_PM_STEPS must be non-negative, got {diagnostic_max_pm_steps}."
        )
diagnostic_snapshots_env = os.environ.get("DISCO_DIAGNOSTIC_SNAPSHOTS", "0").strip()
if diagnostic_snapshots_env:
    diagnostic_snapshot_indices = tuple(
        sorted({int(item.strip()) for item in diagnostic_snapshots_env.split(",") if item.strip()})
    )
else:
    diagnostic_snapshot_indices = (0,)
for _snap_idx in diagnostic_snapshot_indices:
    if _snap_idx < 0 or _snap_idx >= len(z_snaps):
        raise ValueError(
            f"DISCO_DIAGNOSTIC_SNAPSHOTS contains {_snap_idx}, "
            f"but valid snapshot indices are 0..{len(z_snaps) - 1}"
        )

grid_sbox     = int(os.environ.get("DISCO_GRID_SBOX", "8"))
grid          = int(os.environ.get("DISCO_GRID", "192"))
nrand_sel_box = 32768
stats_nsubvols = int(os.environ.get(
    "DISCO_STATS_NSUBVOLS",
    os.environ.get("DISCO_SAVE_NSUBVOLS", "0"),
))
stats_index_mode = os.environ.get(
    "DISCO_STATS_INDEX_MODE",
    os.environ.get("DISCO_SAVE_INDEX_MODE", "first"),
).strip().lower()
stats_index_seed = int(os.environ.get(
    "DISCO_STATS_INDEX_SEED",
    os.environ.get("DISCO_SAVE_INDEX_SEED", "12345"),
))
if diagnostic_sampled and stats_nsubvols <= 0:
    stats_nsubvols = 10000
output_write_chunk_subvols = int(os.environ.get("DISCO_OUTPUT_WRITE_CHUNK_SUBVOLS", "8192"))
if stats_nsubvols < 0:
    raise ValueError(f"DISCO_STATS_NSUBVOLS must be non-negative, got {stats_nsubvols}")
if stats_index_mode not in {"first", "random"}:
    raise ValueError(
        "DISCO_STATS_INDEX_MODE must be 'first' or 'random', "
        f"got {stats_index_mode!r}"
    )
if output_write_chunk_subvols <= 0:
    raise ValueError(
        "DISCO_OUTPUT_WRITE_CHUNK_SUBVOLS must be positive, "
        f"got {output_write_chunk_subvols}"
    )

dtype     = jnp.float32
dtype_num = 32
dtype_c_num = 64

cosmo_params_path = os.environ.get("DISCO_COSMO_PARAMS_PATH", "").strip()
cosmo_source = "hardcoded_fiducial"
if cosmo_params_path:
    cosmo_all = np.loadtxt(cosmo_params_path, dtype=np.float64)
    if cosmo_all.size < 5:
        raise ValueError(
            f"DISCO_COSMO_PARAMS_PATH={cosmo_params_path!r} must contain at least "
            "five values: Omega_m Omega_b h n_s sigma8"
        )
    Om, Ob, h, ns, sigma8 = [float(x) for x in np.ravel(cosmo_all)[:5]]
    cosmo_source = cosmo_params_path
else:
    Om = float(os.environ.get("DISCO_OMEGA_M", "0.3175"))
    Ob = float(os.environ.get("DISCO_OMEGA_B", "0.049"))
    h = float(os.environ.get("DISCO_H", "0.6711"))
    ns = float(os.environ.get("DISCO_NS", "0.9624"))
    sigma8 = float(os.environ.get("DISCO_SIGMA8", "0.834"))
    if any(
        name in os.environ
        for name in ("DISCO_OMEGA_M", "DISCO_OMEGA_B", "DISCO_H", "DISCO_NS", "DISCO_SIGMA8")
    ):
        cosmo_source = "environment"
cosmo = dict(Omega_c=Om - Ob, Omega_b=Ob, h=h, n_s=ns, sigma8=sigma8)

reference_boxsize = float(os.environ.get("DISCO_REFERENCE_BOXSIZE", "1000.0"))
reference_res = int(os.environ.get("DISCO_REFERENCE_RES", "512"))
reference_factor = int(os.environ.get("DISCO_REFERENCE_PM_FACTOR", "2"))
reference_grid = int(os.environ.get("DISCO_REFERENCE_GRID", "64"))
reference_grid_sbox = int(os.environ.get("DISCO_REFERENCE_GRID_SBOX", "8"))
reference_res_pm = reference_factor * reference_res


def _resolution_metrics(boxsize_val: float, res_val: int, res_pm_val: int,
                        grid_val: int, grid_sbox_val: int) -> dict[str, float]:
    grid_tot = grid_val * grid_sbox_val
    return {
        "particle_spacing": boxsize_val / res_val,
        "pm_cell_size": boxsize_val / res_pm_val,
        "output_voxel_size": boxsize_val / grid_tot,
    }


current_resolution = _resolution_metrics(boxsize, res, res_pm, grid, grid_sbox)
reference_resolution = _resolution_metrics(
    reference_boxsize, reference_res, reference_res_pm,
    reference_grid, reference_grid_sbox,
)
resolution_mismatches = [
    key for key in current_resolution
    if not np.isclose(current_resolution[key], reference_resolution[key], rtol=1e-6, atol=1e-8)
]
if resolution_mismatches and not _get_env_bool("DISCO_ALLOW_RESOLUTION_MISMATCH", False):
    details = ", ".join(
        f"{key}: current={current_resolution[key]:.6g}, "
        f"reference={reference_resolution[key]:.6g}"
        for key in resolution_mismatches
    )
    raise ValueError(
        "3Gpc PM setup does not match the 1Gpc reference physical resolution. "
        f"{details}. Set DISCO_ALLOW_RESOLUTION_MISMATCH=1 only for explicit "
        "non-reference tests."
    )


# =============================================================================
# DEVICE MESH
# =============================================================================
def _best_pdims(n_dev: int, res: int):
    """Squarest (a, b) with a*b == n_dev and both a, b divide res."""
    best = None
    for a in range(1, int(n_dev**0.5) + 1):
        if n_dev % a == 0:
            b = n_dev // a
            if res % a == 0 and res % b == 0:
                best = (a, b)   # keep last (most square) valid pair
    if best is None:
        max_suggest = max(32, n_dev + 16)
        valid_counts_4gpu_nodes = []
        for n in range(4, max_suggest + 1, 4):
            for a in range(1, int(n**0.5) + 1):
                if n % a == 0 and res % a == 0 and res % (n // a) == 0:
                    valid_counts_4gpu_nodes.append(n)
                    break
        raise ValueError(f"No valid 2-D mesh for n_dev={n_dev}, res={res}. "
                         f"Choose n_dev=a*b where both a and b divide {res}. "
                         f"For 4-GPU nodes, valid counts up to {max_suggest} are "
                         f"{valid_counts_4gpu_nodes}.")
    return best


def _parse_pdims_override(env_name: str, n_dev: int, res: int, res_pm: int):
    pdims_env = os.environ.get(env_name, "").strip()
    if not pdims_env:
        return None
    try:
        parts = tuple(int(s.strip()) for s in pdims_env.split(","))
    except ValueError as exc:
        raise ValueError(f"{env_name} must be formatted as 'px,py', got {pdims_env!r}.") from exc
    if len(parts) != 2:
        raise ValueError(f"{env_name} must be formatted as 'px,py', got {pdims_env!r}.")
    px, py = parts
    if px <= 0 or py <= 0:
        raise ValueError(f"{env_name} entries must be positive, got {parts}.")
    if px * py != n_dev:
        raise ValueError(f"{env_name}={parts} product must equal n_dev={n_dev}.")
    if res % px != 0 or res % py != 0:
        raise ValueError(f"res={res} must be divisible by {env_name}={parts}.")
    if res_pm % px != 0 or res_pm % py != 0:
        raise ValueError(f"res_pm={res_pm} must be divisible by {env_name}={parts}.")
    return parts


def _choose_lpt_pdims(n_dev: int, res: int, res_pm: int, pm_backend: str):
    override = (
        _parse_pdims_override("DISCO_LPT_PDIMS", n_dev, res, res_pm)
        or _parse_pdims_override("DISCO_PDIMS", n_dev, res, res_pm)
    )
    if override is not None:
        return override
    del pm_backend
    return (1, 1) if n_dev == 1 else _best_pdims(n_dev, res)


def _choose_pm_pdims(n_dev: int, res: int, res_pm: int, pm_backend: str, lpt_pdims):
    override = (
        _parse_pdims_override("DISCO_PM_PDIMS", n_dev, res, res_pm)
        or _parse_pdims_override("DISCO_PDIMS", n_dev, res, res_pm)
    )
    if override is not None:
        return override
    if pm_backend == "JAX_RFFT" and n_dev > 1:
        # The slab RFFT path keeps the half-spectrum axis unsharded.  A square
        # pencil such as (4,4) would have to split N/2+1, which is 1537 for
        # the production run and is not divisible by 4.
        if res % n_dev == 0 and res_pm % n_dev == 0:
            return (n_dev, 1)
        raise ValueError(
            f"DISCO_PM_FFT_BACKEND=JAX_RFFT requires slab pdims={(n_dev, 1)}, "
            f"but res={res} or res_pm={res_pm} is not divisible by n_dev={n_dev}."
        )
    return lpt_pdims

devices = jax.devices()
device  = "gpu" if any(d.platform == "gpu" for d in devices) else "cpu"
n_dev   = jax.device_count()
local_n_dev = jax.local_device_count()
process_id = jax.process_index()
process_count = jax.process_count()
is_leader = process_id == 0
discodj_device = None if process_count > 1 else device
pdims   = _choose_lpt_pdims(n_dev, res, res_pm, pm_fft_backend)
pm_pdims = _choose_pm_pdims(n_dev, res, res_pm, pm_fft_backend, pdims)
distributed_cpu_postprocess = _get_env_bool(
    "DISCO_DISTRIBUTED_CPU_POSTPROCESS",
    process_count > 1,
)
cpu_reduce_backend = os.environ.get("DISCO_CPU_REDUCE_BACKEND", "socket").strip().lower()


def log(message: str, *, leader_only: bool = False) -> None:
    if leader_only and not is_leader:
        return
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"[proc {process_id}/{process_count}] {message}",
        flush=True,
    )


print(
    f"device={device}, global_n_dev={n_dev}, local_n_dev={local_n_dev}, "
    f"process={process_id}/{process_count}, lpt_pdims={pdims}, pm_pdims={pm_pdims}",
    flush=True,
)
print(
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}, "
    f"SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS', 'unset')}",
    flush=True,
)
print(
    f"boxsize={boxsize}, res={res}, factor={factor}, res_pm={res_pm}, "
    f"grid={grid}, grid_sbox={grid_sbox}, halo_size={halo_size}, "
    f"lpt_order={n_order}",
    flush=True,
)
print(
    "physical resolution match to 1Gpc reference: "
    f"particle_spacing={current_resolution['particle_spacing']:.6f} Mpc/h "
    f"(ref {reference_resolution['particle_spacing']:.6f}), "
    f"pm_cell_size={current_resolution['pm_cell_size']:.6f} Mpc/h "
    f"(ref {reference_resolution['pm_cell_size']:.6f}), "
    f"output_voxel_size={current_resolution['output_voxel_size']:.6f} Mpc/h "
    f"(ref {reference_resolution['output_voxel_size']:.6f})",
    flush=True,
)
print(
    f"IC FFT backend={ic_fft_backend}, LPT FFT backend={lpt_fft_backend}, "
    f"LPT mu2 FFT backend={lpt_mu2_fft_backend}, "
    f"PM FFT backend={pm_fft_backend}, "
    f"LPT grad_order={lpt_grad_kernel_order}, PM grad_order={grad_kernel_order}, "
    f"distributed_cpu_postprocess={distributed_cpu_postprocess}, "
    f"cpu_reduce_backend={cpu_reduce_backend}, "
    f"jaxdecomp transpose_backend={jaxdecomp.config.transpose_comm_backend}, "
    f"halo_backend={jaxdecomp.config.halo_comm_backend}, "
    f"axis_contiguous={jaxdecomp.config.transpose_axis_contiguous}",
    flush=True,
)
pm_cell_size = boxsize / res_pm
halo_usable_cells = halo_size / 2.0 - (worder // 2) - halo_safety_cells
if halo_usable_cells <= 0:
    raise ValueError(
        f"DISCO_HALO_SIZE={halo_size} is too small for worder={worder} "
        f"and DISCO_HALO_SAFETY_CELLS={halo_safety_cells}."
    )
print(
    f"DISCO_OUTPUT_TAG={output_tag}, DISCO_VALIDATE_HALO={int(validate_halo)}, "
    f"DISCO_DIAGNOSTIC_MODE={diagnostic_mode or 'off'}, "
    f"DISCO_DIAGNOSTIC_SNAPSHOTS={diagnostic_snapshot_indices}, "
    f"DISCO_DIAGNOSTIC_MAX_PM_STEPS={diagnostic_max_pm_steps}, "
    f"DISCO_DIAGNOSTIC_STEP_STATS={int(diagnostic_step_stats)}, "
    f"DISCO_STATS_NSUBVOLS={stats_nsubvols}, "
    f"DISCO_STATS_INDEX_MODE={stats_index_mode}, "
    f"DISCO_OUTPUT_WRITE_CHUNK_SUBVOLS={output_write_chunk_subvols}, "
    f"DISCO_HALO_SAFETY_CELLS={halo_safety_cells:.1f}, "
    f"pm_cell_size={pm_cell_size:.6f} Mpc/h, "
    f"validated usable halo={halo_usable_cells:.1f} PM cells "
    f"({halo_usable_cells * pm_cell_size:.2f} Mpc/h)",
    flush=True,
)

mesh_devices   = create_device_mesh(pdims)
mesh           = Mesh(mesh_devices, axis_names=('x', 'y'))
sharding_disp  = NamedSharding(mesh, P('x', 'y', None, None))  # LPT/IC (Lx,Ly,Lz,3)
sharding_field = NamedSharding(mesh, P('x', 'y', None))         # LPT/IC (Lx,Ly,Lz)

if pm_pdims == pdims:
    pm_mesh = mesh
    pm_axis_names = ('x', 'y')
else:
    if pm_fft_backend != "JAX_RFFT":
        raise ValueError(
            "Separate LPT and PM mesh shapes are only supported for "
            "DISCO_PM_FFT_BACKEND=JAX_RFFT.  jaxDecomp full FFT backends "
            "should keep DISCO_PM_PDIMS equal to DISCO_LPT_PDIMS."
        )
    pm_axis_names = ('pm_x', 'pm_y')
    pm_mesh_devices = create_device_mesh(pm_pdims)
    pm_mesh = Mesh(pm_mesh_devices, axis_names=pm_axis_names)
sharding_pm_disp = NamedSharding(pm_mesh, P(*pm_axis_names, None, None))
needs_pm_reshard = pdims != pm_pdims


@partial(jax.jit, donate_argnums=(0, 1), out_shardings=(sharding_pm_disp, sharding_pm_disp))
def _reshard_lpt_state_to_pm_mesh(psi, mom):
    return (
        lax.with_sharding_constraint(psi, sharding_pm_disp),
        lax.with_sharding_constraint(mom, sharding_pm_disp),
    )


def _host_array_to_global_sharded(host_arr: np.ndarray, sharding: NamedSharding,
                                  dtype) -> jax.Array:
    """Create a global sharded JAX array from local host slices.

    `jax.device_put(host_arr, sharding)` is dangerous for large multihost
    inputs: JAX first checks that all hosts have the same value via a
    process_allgather, which can temporarily replicate the full IC once per
    process.  For the 1536^3 IC this is ~13.5 GiB per process, so a 4-node run
    tries to allocate ~54 GiB before useful sharding even starts.  The callback
    path copies only the addressable shard slices owned by this process.
    """
    global_shape = tuple(int(s) for s in host_arr.shape)
    np_dtype = np.dtype(dtype)

    def _slice_callback(index):
        shard = host_arr[index]
        if shard.dtype != np_dtype:
            shard = shard.astype(np_dtype, copy=False)
        return np.ascontiguousarray(shard)

    return jax.make_array_from_callback(global_shape, sharding, _slice_callback)


def _preflight_fft_backend(backend: str) -> None:
    if backend.lower() != "cudecomp":
        return
    if device != "gpu":
        log(f"Skipping {backend} FFT preflight on non-GPU platform")
        return
    if pdims[0] * pdims[1] != process_count:
        raise RuntimeError(
            "DISCO_LPT_FFT_BACKEND=cudecomp requires one MPI/JAX process per "
            "GPU shard because cuDecomp requires product(pdims) to equal the "
            "MPI rank count. Current launch has "
            f"global_n_dev={n_dev}, local_n_dev={local_n_dev}, "
            f"process_count={process_count}, pdims={pdims}. "
            "Launch with one task per GPU, set CUDA_VISIBLE_DEVICES to one GPU "
            "per local rank, and set JAX_LOCAL_DEVICE_IDS=0; or use "
            "DISCO_LPT_FFT_BACKEND=JAX for one-process-per-node runs."
        )

    test_n = max(32, 4 * max(pdims))
    while test_n % pdims[0] != 0 or test_n % pdims[1] != 0:
        test_n += 1

    host = np.ones((test_n, test_n, test_n), dtype=np.complex64)
    arr = _host_array_to_global_sharded(host, sharding_field, dtype=np.complex64)
    expected = np.array(multihost_utils.process_allgather(arr, tiled=True))

    @jax.jit
    def _roundtrip(x):
        k = jaxdecomp.pfft3d(x, norm="backward", backend=backend)
        y = jaxdecomp.pifft3d(k, norm="backward", backend=backend)
        return y

    log(f"Preflighting jaxDecomp FFT backend={backend} with shape={host.shape}")
    try:
        out = _roundtrip(arr)
        out.block_until_ready()
        actual = np.array(multihost_utils.process_allgather(out, tiled=True))
        max_err = float(np.max(np.abs(actual - expected)))
    except Exception as exc:
        message = str(exc)
        if "compiled without CUDA support" in message or "cuDecomp functions are not supported" in message:
            raise RuntimeError(
                "DISCO_LPT_FFT_BACKEND=cudecomp was requested, but the installed "
                "jaxDecomp extension was compiled without CUDA/cuDecomp support. "
                "Use DISCO_LPT_FFT_BACKEND=JAX with the current environment, or "
                "install a CUDA-enabled jaxDecomp build before requesting cudecomp."
            ) from exc
        raise
    if max_err > 1e-5:
        raise RuntimeError(
            f"Preflight jaxDecomp FFT backend={backend} failed numerical "
            f"roundtrip check for shape={host.shape}: max_err={max_err}. "
            "Refusing to run LPT with this backend because it would corrupt the "
            "physics. Use DISCO_LPT_FFT_BACKEND=JAX until the cuDecomp backend "
            "passes this check in the current environment."
        )
    log(f"Preflight jaxDecomp FFT backend={backend} passed with max_err={max_err:.3e}")


def _preflight_pm_rfft_backend() -> None:
    if pm_fft_backend != "JAX_RFFT":
        return
    if pm_pdims[1] != 1:
        raise RuntimeError(
            f"PM JAX_RFFT backend requires slab pdims=(n,1), got {pm_pdims}."
        )
    test_n = max(32, 2 * pm_pdims[0])
    while test_n % pm_pdims[0] != 0:
        test_n += 1
    test_halo = 4

    psi0 = jax.device_put(
        jnp.zeros((test_n, test_n, test_n, 3), dtype=dtype),
        sharding_pm_disp,
    )
    mom0 = jax.device_put(
        jnp.zeros((test_n, test_n, test_n, 3), dtype=dtype),
        sharding_pm_disp,
    )

    @jax.jit
    def _go(psi, mom):
        return kick_PM_distributed(
            psi,
            mom,
            alpha=jnp.asarray(1.0, dtype=dtype),
            beta=jnp.asarray(0.1, dtype=dtype),
            dim=dim,
            res_pm=test_n,
            boxsize=float(test_n),
            halo_size=test_halo,
            sharding=sharding_pm_disp,
            grad_order=0,
            lap_order=0,
            dtype_num=dtype_num,
            worder=worder,
            deconvolve=False,
            fft_backend=pm_fft_backend,
        )

    log(f"Preflighting PM slab RFFT backend with shape={(test_n, test_n, test_n)}")
    out = _go(psi0, mom0)
    out.block_until_ready()
    max_abs = float(jnp.max(jnp.abs(out)).block_until_ready())
    if max_abs > 1e-5:
        raise RuntimeError(
            f"PM slab RFFT preflight failed: zero-displacement kick has max_abs={max_abs}."
        )
    log(f"Preflight PM slab RFFT backend passed with max_abs={max_abs:.3e}")


@jax.jit
def _state_stats_jax(psi, mom, F_end, a_end, pm_cell):
    """Small global summaries of PM state without host particle gather/painting."""
    vel = mom * (F_end / a_end * 100.0)
    psi_cells = psi / pm_cell
    axes = (0, 1, 2)

    def _component_stats(arr):
        return jnp.stack(
            [
                jnp.mean(arr, axis=axes),
                jnp.std(arr, axis=axes),
                jnp.sqrt(jnp.mean(arr * arr, axis=axes)),
                jnp.min(arr, axis=axes),
                jnp.max(arr, axis=axes),
                jnp.max(jnp.abs(arr), axis=axes),
            ],
            axis=0,
        )

    def _magnitude_stats(arr):
        mag2 = jnp.sum(arr * arr, axis=-1)
        return jnp.asarray(
            [
                jnp.sqrt(jnp.mean(mag2)),
                jnp.mean(jnp.sqrt(mag2)),
                jnp.sqrt(jnp.max(mag2)),
            ],
            dtype=arr.dtype,
        )

    comp = jnp.stack(
        [
            _component_stats(psi_cells),
            _component_stats(mom),
            _component_stats(vel),
        ],
        axis=0,
    )
    mag = jnp.stack(
        [
            _magnitude_stats(psi_cells),
            _magnitude_stats(mom),
            _magnitude_stats(vel),
        ],
        axis=0,
    )
    return comp, mag


def _collect_state_stats(psi, mom, snapshot: int, z: float, a_end: float,
                         F_end: float) -> dict:
    comp, mag = _state_stats_jax(
        psi,
        mom,
        jnp.asarray(F_end, dtype=dtype),
        jnp.asarray(a_end, dtype=dtype),
        jnp.asarray(pm_cell_size, dtype=dtype),
    )
    comp_np = np.asarray(comp.block_until_ready(), dtype=np.float64)
    mag_np = np.asarray(mag.block_until_ready(), dtype=np.float64)
    labels = ("psi_pm_cells", "mom_D_units", "vel_km_s")
    stat_names = ("mean", "std", "rms", "min", "max", "max_abs")
    mag_names = ("rms_mag", "mean_mag", "max_mag")
    stats = {
        "snapshot": int(snapshot),
        "z": float(z),
        "a_end": float(a_end),
        "F_end": float(F_end),
        "labels": labels,
        "component_stat_names": stat_names,
        "magnitude_stat_names": mag_names,
        "component_stats": comp_np,
        "magnitude_stats": mag_np,
    }
    if is_leader:
        for idx, label in enumerate(labels):
            rms = comp_np[idx, stat_names.index("rms")]
            max_abs = comp_np[idx, stat_names.index("max_abs")]
            mag_vals = mag_np[idx]
            log(
                f"Snapshot {snapshot}: state_stats {label} "
                f"rms_xyz={rms[0]:.6g},{rms[1]:.6g},{rms[2]:.6g} "
                f"maxabs_xyz={max_abs[0]:.6g},{max_abs[1]:.6g},{max_abs[2]:.6g} "
                f"mag_rms/mean/max={mag_vals[0]:.6g}/{mag_vals[1]:.6g}/{mag_vals[2]:.6g}",
                leader_only=True,
            )
    return stats


@partial(jax.jit, static_argnames=("axis",))
def _spectral_psi1_rms_component(fphi, axis: int):
    """Parseval RMS of the first-order displacement component from fphi."""
    k_vecs = build_k_vecs_dist(fphi, boxsize=boxsize, res=res)
    deriv = gradient_kernel_dist(k_vecs, axis=axis, order=lpt_grad_kernel_order)
    psi1_k = -deriv * fphi
    norm = jnp.asarray(float(res) ** 6, dtype=psi1_k.real.dtype)
    return jnp.sqrt(jnp.sum(jnp.abs(psi1_k) ** 2) / norm)


def _collect_spectral_psi1_rms(fphi) -> np.ndarray:
    """Return z=0 Zel'dovich component RMS implied by fphi, in Mpc/h."""
    vals = []
    for axis in range(3):
        val = _spectral_psi1_rms_component(fphi, axis=axis)
        vals.append(float(np.asarray(val.block_until_ready())))
    out = np.asarray(vals, dtype=np.float64)
    if is_leader:
        log(
            "Initial fphi spectral psi1 rms_xyz="
            f"{out[0]:.6g},{out[1]:.6g},{out[2]:.6g} Mpc/h",
            leader_only=True,
        )
    return out


@partial(jax.jit, static_argnames=("axis",))
def _direct_psi1_component_stats(fphi, axis: int):
    """Real-space stats for psi1 component via the configured LPT inverse FFT."""
    k_vecs = build_k_vecs_dist(fphi, boxsize=boxsize, res=res)
    deriv = gradient_kernel_dist(k_vecs, axis=axis, order=lpt_grad_kernel_order)
    comp_k = -deriv * fphi
    comp = jaxdecomp.pifft3d(comp_k, norm="backward", backend=lpt_fft_backend).real.astype(dtype)
    return jnp.asarray(
        [
            jnp.mean(comp),
            jnp.std(comp),
            jnp.sqrt(jnp.mean(comp * comp)),
            jnp.min(comp),
            jnp.max(comp),
            jnp.max(jnp.abs(comp)),
        ],
        dtype=comp.dtype,
    )


def _collect_direct_psi1_stats(fphi) -> np.ndarray:
    """Component stats for z=0 Zel'dovich displacement from direct inverse FFT."""
    rows = []
    for axis in range(3):
        stats = _direct_psi1_component_stats(fphi, axis=axis)
        rows.append(np.asarray(stats.block_until_ready(), dtype=np.float64))
    out = np.stack(rows, axis=-1)
    if is_leader:
        rms = out[2]
        max_abs = out[5]
        log(
            "Initial direct-iFFT psi1 rms_xyz="
            f"{rms[0]:.6g},{rms[1]:.6g},{rms[2]:.6g} Mpc/h; "
            f"maxabs_xyz={max_abs[0]:.6g},{max_abs[1]:.6g},{max_abs[2]:.6g}",
            leader_only=True,
        )
    return out


def _compute_1lpt_initial_state_from_fphi(fphi, Dplus: float):
    """Fast first-order initial state: psi=D*psi1 and mom=dpsi/dD=psi1."""
    fft_sharding = getattr(fphi, "sharding", None)

    def _component(axis: int):
        @partial(jax.jit, static_argnames=("axis",))
        def _compute(phi, *, axis: int):
            k_vecs = build_k_vecs_dist(phi, boxsize=boxsize, res=res)
            deriv = gradient_kernel_dist(k_vecs, axis=axis, order=lpt_grad_kernel_order)
            comp_k = -deriv * phi
            if fft_sharding is not None:
                comp_k = lax.with_sharding_constraint(comp_k, fft_sharding)
            comp = jaxdecomp.pifft3d(comp_k, norm="backward", backend=lpt_fft_backend).real.astype(dtype)
            return lax.with_sharding_constraint(comp, sharding_field)

        return _compute(fphi, axis=axis)

    D = jnp.asarray(Dplus, dtype=dtype)
    psi1_components = []
    for axis in range(3):
        log(f"1LPT component axis={axis}: inverse FFT start")
        comp = _component(axis)
        comp.block_until_ready()
        log(f"1LPT component axis={axis}: inverse FFT done")
        psi1_components.append(comp)

    @jax.jit
    def _stack(c0, c1, c2):
        mom = jnp.stack([c0, c1, c2], axis=-1)
        psi = D * mom
        return (
            lax.with_sharding_constraint(psi.astype(dtype), sharding_disp),
            lax.with_sharding_constraint(mom.astype(dtype), sharding_disp),
        )

    psi, mom = _stack(*psi1_components)
    psi.block_until_ready()
    mom.block_until_ready()
    return psi, mom


# =============================================================================
# DISTRIBUTED fphi BUILDER
# =============================================================================
# discodj_dist.with_external_ics(delta=..., sharding=...) calls
# `inv_laplace_kernel(k_vecs)` outside any jit context: ksquare = sum(ki^2)
# eagerly broadcasts the sparse 1-D k_vecs into a full (res,res,res) array on
# a single device, which OOMs at res=1536 (~14.5 GB float32 + ~14.5 GB kernel
# on GPU 0).  We replicate that math inside jax.jit with explicit sharding
# constraints so XLA partitions ksquare across devices, then inject the
# sharded fphi_full into dj_dist via update(ics=...).
@partial(jax.jit, donate_argnums=(0,))
def _compute_fphi_full_sharded(delta):
    fdelta = jaxdecomp.pfft3d(delta.astype(jnp.complex64), norm="backward",
                              backend=ic_fft_backend)
    fdelta = lax.with_sharding_constraint(fdelta, sharding_field)
    k_vecs = build_k_vecs_dist(fdelta, boxsize=boxsize, res=res)
    ksquare = sum(ki ** 2 for ki in k_vecs)
    mask = (ksquare != 0).astype(jnp.float32)
    ksquare = ksquare.at[0, 0, 0].set(jnp.float32(1.0))
    ksquare = lax.with_sharding_constraint(ksquare, sharding_field)
    inv_lap = -1.0 / ksquare
    inv_lap = inv_lap * mask
    fphi = inv_lap * fdelta
    return lax.with_sharding_constraint(fphi, sharding_field)


# Safety belt for with_lpt: inside compute_lpt_distributed,
# inv_laplace_kernel(k_vecs) builds ksquare = sum(ki**2) which broadcasts the
# sparse 1-D k_vecs into a full (res,res,res) array.  At res=1536 that's a
# ~14.5 GB float32 intermediate.  It runs inside jit, so XLA/GSPMD *should*
# propagate the downstream _with_sharding annotations backward, but if it
# doesn't the result is a single-device OOM identical to the with_external_ics
# crash.  Wrap inv_laplace_kernel to emit an explicitly sharded output
# matching our fphi_full sharding.  (gradient_kernel_dist returns sparse 1-D
# shapes like (N,1,1) — leave those alone; they don't participate in the
# large unsharded broadcast.)
_orig_inv_lap = _nlpt_dist.inv_laplace_kernel
def _sharded_inv_lap(k_vecs, *a, **kw):
    return lax.with_sharding_constraint(
        _orig_inv_lap(k_vecs, *a, **kw), sharding_field)
_nlpt_dist.inv_laplace_kernel = _sharded_inv_lap


# =============================================================================
# NUMBA HELPERS  (verbatim from pm.py)
# =============================================================================
@njit(parallel=True, fastmath=True, cache=True)
def _get_padded_mat_numba_core(Npart_pad, n_pad, grid_sbox, grid, fac):
    fac3_inv = np.float32(1.0 / (fac * fac * fac))
    result   = np.zeros((grid, grid, grid, grid_sbox, grid_sbox, grid_sbox), dtype=np.float32)
    for idx in prange(grid * grid * grid):
        gi = idx // (grid * grid)
        gj = (idx // grid) % grid
        gk = idx % grid
        i0 = gi * grid_sbox
        j0 = gj * grid_sbox
        k0 = gk * grid_sbox
        for si in range(grid_sbox):
            i_base = i0 + si * fac
            for sj in range(grid_sbox):
                j_base = j0 + sj * fac
                for sk in range(grid_sbox):
                    k_base = k0 + sk * fac
                    total  = np.float32(0.0)
                    for di in range(fac):
                        for dj in range(fac):
                            for dk in range(fac):
                                total += Npart_pad[i_base + di, j_base + dj, k_base + dk]
                    result[gi, gj, gk, si, sj, sk] = total * fac3_inv
    return result


def get_padded_mat_numba(Npart, n_pad, grid_sbox, grid):
    Npart_pad = np.pad(Npart, n_pad, mode='wrap').astype(np.float32)
    box_size  = grid_sbox + 2 * n_pad
    fac       = box_size // grid_sbox
    result    = _get_padded_mat_numba_core(Npart_pad, n_pad, grid_sbox, grid, fac)
    return result, None


@njit(parallel=True, fastmath=True, cache=True)
def _extract_padded_subvolumes_numba_core(Npart_pad, subvol_indices, n_pad,
                                          grid_sbox, grid, fac):
    fac3_inv = np.float32(1.0 / (fac * fac * fac))
    n_subvols = subvol_indices.shape[0]
    out = np.zeros((n_subvols, grid_sbox, grid_sbox, grid_sbox), dtype=np.float32)
    for out_idx in prange(n_subvols):
        subvol_idx = subvol_indices[out_idx]
        gi = subvol_idx // (grid * grid)
        gj = (subvol_idx // grid) % grid
        gk = subvol_idx % grid
        i0 = gi * grid_sbox
        j0 = gj * grid_sbox
        k0 = gk * grid_sbox
        for si in range(grid_sbox):
            i_base = i0 + si * fac
            for sj in range(grid_sbox):
                j_base = j0 + sj * fac
                for sk in range(grid_sbox):
                    k_base = k0 + sk * fac
                    total = np.float32(0.0)
                    for di in range(fac):
                        for dj in range(fac):
                            for dk in range(fac):
                                total += Npart_pad[
                                    i_base + di,
                                    j_base + dj,
                                    k_base + dk,
                                ]
                    out[out_idx, si, sj, sk] = total * fac3_inv
    return out


def extract_padded_subvolumes_numba(Npart, subvol_indices, n_pad, grid_sbox, grid):
    subvol_indices = np.asarray(subvol_indices, dtype=np.int64)
    Npart_pad = np.pad(Npart, n_pad, mode='wrap').astype(np.float32)
    box_size = grid_sbox + 2 * n_pad
    fac = box_size // grid_sbox
    return _extract_padded_subvolumes_numba_core(
        Npart_pad, subvol_indices, n_pad, grid_sbox, grid, fac
    )


def mat_reshape_fast(mat, grid, grid_sbox):
    if mat.ndim == 3:
        return mat.reshape(grid, grid_sbox, grid, grid_sbox, grid, grid_sbox).transpose(0, 2, 4, 1, 3, 5)
    extra = mat.shape[3:]
    return (mat.reshape(grid, grid_sbox, grid, grid_sbox, grid, grid_sbox, *extra)
               .transpose(0, 2, 4, 1, 3, 5, *range(6, 6 + len(extra))))


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def _make_subvol_selection(n_subvols: int, mode: str, seed: int) -> np.ndarray | None:
    if n_subvols <= 0:
        return None
    n_save = min(n_subvols, grid**3)
    if mode == "first":
        return np.arange(n_save, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(grid**3, size=n_save, replace=False)).astype(np.int64, copy=False)


def _write_snapshot_fields_to_memmap(out_mm: np.memmap, fields: np.ndarray,
                                     channel_offset: int, chunk_subvols: int,
                                     label: str) -> None:
    """Write one full snapshot field block into the final float16 .npy file."""
    n_subvols = fields.shape[0]
    n_channels = fields.shape[-1]
    n_chunks = (n_subvols + chunk_subvols - 1) // chunk_subvols
    log(
        f"{label}: streaming full field to output channels "
        f"{channel_offset}:{channel_offset + n_channels}, "
        f"chunk_subvols={chunk_subvols}, n_chunks={n_chunks}, RSS={_rss_gb():.1f} GB",
        leader_only=True,
    )
    t0 = time.perf_counter()
    for chunk_idx, i0 in enumerate(range(0, n_subvols, chunk_subvols)):
        i1 = min(n_subvols, i0 + chunk_subvols)
        out_mm[i0:i1, ..., channel_offset:channel_offset + n_channels] = (
            fields[i0:i1].astype(np.float16)
        )
        if chunk_idx == 0 or chunk_idx == n_chunks - 1 or (chunk_idx + 1) % 64 == 0:
            log(
                f"{label}: wrote output chunk {chunk_idx + 1}/{n_chunks} "
                f"subvols {i0}:{i1}",
                leader_only=True,
            )
    out_mm.flush()
    log(
        f"{label}: full-field streaming write done in {time.perf_counter() - t0:.1f} s",
        leader_only=True,
    )


def _extract_subvolumes_from_grid(mat: np.ndarray, subvol_indices: np.ndarray,
                                  grid: int, grid_sbox: int,
                                  out: np.ndarray | None = None) -> np.ndarray:
    """Extract selected grid_sbox^3 blocks without materializing all subvolumes."""
    subvol_indices = np.asarray(subvol_indices, dtype=np.int64)
    if out is None:
        out = np.empty((len(subvol_indices), grid_sbox, grid_sbox, grid_sbox), dtype=mat.dtype)
    elif out.shape != (len(subvol_indices), grid_sbox, grid_sbox, grid_sbox):
        raise ValueError(f"Unexpected output shape {out.shape} for {len(subvol_indices)} subvolumes")

    ix, iy, iz = np.unravel_index(subvol_indices, (grid, grid, grid))
    for out_idx, (i, j, k) in enumerate(zip(ix, iy, iz)):
        x0 = int(i) * grid_sbox
        y0 = int(j) * grid_sbox
        z0 = int(k) * grid_sbox
        out[out_idx] = mat[
            x0:x0 + grid_sbox,
            y0:y0 + grid_sbox,
            z0:z0 + grid_sbox,
        ]
    return out


_SOCKET_REDUCE_COUNTER = 0
_SOCKET_CHUNK_HEADER = struct.Struct("!iiiiq")


def _get_postprocess_backend():
    if not distributed_cpu_postprocess:
        return "serial", None
    if process_count <= 1:
        return "serial", None
    backend = cpu_reduce_backend
    if backend in {"", "auto"}:
        backend = "socket"
    if backend == "socket":
        return "socket", None
    if backend != "mpi":
        raise ValueError(
            "DISCO_CPU_REDUCE_BACKEND must be 'socket', 'mpi', or 'auto', "
            f"got {cpu_reduce_backend!r}."
        )

    global MPI
    if MPI is None:
        try:
            from mpi4py import MPI as _MPI
        except ModuleNotFoundError:
            raise RuntimeError(
                "DISCO_CPU_REDUCE_BACKEND=mpi requires mpi4py in the runtime "
                "environment. Use DISCO_CPU_REDUCE_BACKEND=socket with the "
                "current launcher, or install mpi4py against the same MPI used "
                "by mpirun."
            )
        MPI = _MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    if mpi_rank != process_id or mpi_size != process_count:
        raise RuntimeError(
            "MPI/JAX rank mismatch in distributed CPU postprocess: "
            f"MPI rank/size={mpi_rank}/{mpi_size}, "
            f"JAX process index/count={process_id}/{process_count}. "
            "The Slurm launcher must use one MPI rank for each JAX process."
        )
    return "mpi", comm


def _socket_reduce_host_and_port(reduce_id: int) -> tuple[str, int]:
    host_port = os.environ.get("JAX_COORDINATOR_ADDRESS") or _get_slurm_coordinator_address()
    host = host_port.rsplit(":", 1)[0] if ":" in host_port else host_port
    if host in {"", "0.0.0.0", "::"}:
        host = os.environ.get("JAX_COORDINATOR_HOST", "127.0.0.1")
    base_port_env = os.environ.get("DISCO_CPU_REDUCE_PORT")
    if base_port_env:
        base_port = int(base_port_env)
    else:
        coord_port = int(os.environ.get("JAX_COORDINATOR_PORT", "12355"))
        base_port = coord_port + 1000
    return host, base_port + reduce_id


def _open_socket_reducer_server(preferred_port: int, label: str, reduce_id: int):
    port_span = int(os.environ.get("DISCO_CPU_REDUCE_PORT_SPAN", "2000"))
    last_error = None
    for offset in range(max(1, port_span)):
        port = preferred_port + offset
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("", port))
            server.listen(max(1, process_count - 1))
            if offset:
                log(
                    f"{label}: preferred reducer port {preferred_port} was busy; "
                    f"using {port} for reduce id={reduce_id}",
                    leader_only=True,
                )
            return server, port
        except OSError as exc:
            server.close()
            last_error = exc
            if exc.errno not in (errno.EADDRINUSE, errno.EADDRNOTAVAIL):
                raise
    raise RuntimeError(
        f"{label}: could not bind any reducer port in "
        f"[{preferred_port}, {preferred_port + max(1, port_span) - 1}]"
    ) from last_error


def _broadcast_socket_reducer_port(chosen_port: int, label: str, reduce_id: int) -> int:
    ports = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(chosen_port, dtype=np.int32),
            tiled=False,
        ),
        dtype=np.int32,
    ).reshape(-1)
    nonzero = ports[ports > 0]
    if len(nonzero) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one root reducer port for reduce id={reduce_id}, "
            f"got {ports.tolist()}"
        )
    return int(nonzero[0])


def _recv_exact_into(conn: socket.socket, arr: np.ndarray) -> None:
    view = memoryview(arr).cast("B")
    offset = 0
    while offset < len(view):
        n_recv = conn.recv_into(view[offset:])
        if n_recv == 0:
            raise ConnectionError(f"Socket closed after {offset} / {len(view)} bytes")
        offset += n_recv


def _socket_send_grid(host: str, port: int, grid: np.ndarray, label: str) -> None:
    payload = memoryview(grid).cast("B")
    last_error = None
    for _ in range(240):
        try:
            with socket.create_connection((host, port), timeout=10.0) as conn:
                conn.sendall(payload)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"{label}: could not connect to root socket reducer at {host}:{port}"
    ) from last_error


def _socket_send_chunk(host: str, port: int, reduce_id: int, chunk_idx: int,
                       chunk: np.ndarray, label: str) -> None:
    if not chunk.flags.c_contiguous:
        chunk = np.ascontiguousarray(chunk)
    payload = memoryview(chunk).cast("B")
    header = _SOCKET_CHUNK_HEADER.pack(
        int(reduce_id),
        int(chunk_idx),
        int(process_id),
        int(np.dtype(chunk.dtype).itemsize),
        int(len(payload)),
    )
    last_error = None
    for _ in range(240):
        try:
            with socket.create_connection((host, port), timeout=10.0) as conn:
                conn.sendall(header)
                conn.sendall(payload)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"{label}: could not connect to root socket reducer at {host}:{port} "
        f"for chunk {chunk_idx}"
    ) from last_error


def _recv_exact_bytes(conn: socket.socket, nbytes: int) -> bytes:
    buf = bytearray(nbytes)
    view = memoryview(buf)
    offset = 0
    while offset < nbytes:
        n_recv = conn.recv_into(view[offset:])
        if n_recv == 0:
            raise ConnectionError(f"Socket closed after {offset} / {nbytes} header bytes")
        offset += n_recv
    return bytes(buf)


def _socket_reduce_chunk_planes(shape: tuple[int, ...]) -> int:
    plane_bytes = int(np.prod(shape[1:], dtype=np.int64)) * np.dtype(np.float32).itemsize
    explicit_planes = os.environ.get("DISCO_CPU_REDUCE_CHUNK_PLANES", "").strip()
    if explicit_planes:
        return max(1, min(shape[0], int(explicit_planes)))
    chunk_mb = float(os.environ.get("DISCO_CPU_REDUCE_CHUNK_MB", "256"))
    chunk_bytes = max(1, int(chunk_mb * 1024 * 1024))
    return max(1, min(shape[0], chunk_bytes // max(1, plane_bytes)))


def _socket_sum_grid_to_root_inplace(grid_local: np.ndarray, label: str):
    global _SOCKET_REDUCE_COUNTER
    reduce_id = _SOCKET_REDUCE_COUNTER
    _SOCKET_REDUCE_COUNTER += 1
    host, preferred_port = _socket_reduce_host_and_port(reduce_id)
    chunk_planes = _socket_reduce_chunk_planes(grid_local.shape)
    n_chunks = (grid_local.shape[0] + chunk_planes - 1) // chunk_planes
    t0 = time.perf_counter()

    server = None
    if is_leader:
        server, chosen_port = _open_socket_reducer_server(preferred_port, label, reduce_id)
    else:
        chosen_port = 0

    port = _broadcast_socket_reducer_port(chosen_port, label, reduce_id)
    log(
        f"{label}: socket sum-reduce start id={reduce_id}, "
        f"root={host}:{port}, preferred_port={preferred_port}, "
        f"grid={grid_local.shape}, chunk_planes={chunk_planes}, "
        f"n_chunks={n_chunks}, RSS={_rss_gb():.1f} GB"
    )

    multihost_utils.sync_global_devices(f"cpu_socket_reduce_{reduce_id}_listen")

    try:
        for chunk_idx, i0 in enumerate(range(0, grid_local.shape[0], chunk_planes)):
            i1 = min(grid_local.shape[0], i0 + chunk_planes)
            chunk = grid_local[i0:i1]
            multihost_utils.sync_global_devices(
                f"cpu_socket_reduce_{reduce_id}_chunk_{chunk_idx}_start"
            )
            if is_leader:
                recv_chunk = np.empty_like(chunk)
                for _ in range(process_count - 1):
                    conn, addr = server.accept()
                    with conn:
                        header = _recv_exact_bytes(conn, _SOCKET_CHUNK_HEADER.size)
                        got_reduce_id, got_chunk_idx, got_rank, itemsize, nbytes = (
                            _SOCKET_CHUNK_HEADER.unpack(header)
                        )
                        expected_nbytes = int(recv_chunk.nbytes)
                        if (
                            got_reduce_id != reduce_id
                            or got_chunk_idx != chunk_idx
                            or got_rank == process_id
                            or itemsize != np.dtype(grid_local.dtype).itemsize
                            or nbytes != expected_nbytes
                        ):
                            raise RuntimeError(
                                f"{label}: bad socket reducer header from {addr}: "
                                f"reduce_id={got_reduce_id}, chunk={got_chunk_idx}, "
                                f"rank={got_rank}, itemsize={itemsize}, "
                                f"nbytes={nbytes}; expected reduce_id={reduce_id}, "
                                f"chunk={chunk_idx}, nbytes={expected_nbytes}."
                            )
                        _recv_exact_into(conn, recv_chunk)
                    chunk += recv_chunk
                del recv_chunk
                if chunk_idx == 0 or chunk_idx == n_chunks - 1 or (chunk_idx + 1) % 8 == 0:
                    log(
                        f"{label}: socket chunk {chunk_idx + 1}/{n_chunks} reduced "
                        f"({i0}:{i1})",
                        leader_only=True,
                    )
            else:
                _socket_send_chunk(host, port, reduce_id, chunk_idx, chunk, label)
                if chunk_idx == 0 or chunk_idx == n_chunks - 1 or (chunk_idx + 1) % 8 == 0:
                    log(
                        f"{label}: socket chunk {chunk_idx + 1}/{n_chunks} sent "
                        f"({i0}:{i1})"
                    )
            multihost_utils.sync_global_devices(
                f"cpu_socket_reduce_{reduce_id}_chunk_{chunk_idx}_done"
            )
    finally:
        if server is not None:
            server.close()

    out = grid_local if is_leader else None
    log(
        f"{label}: socket sum-reduce {'done' if is_leader else 'send done'} "
        f"in {time.perf_counter() - t0:.1f} s"
    )
    multihost_utils.sync_global_devices(f"cpu_socket_reduce_{reduce_id}_done")
    return out


def _mpi_sum_grid_to_root_inplace(grid_local: np.ndarray, comm, label: str):
    """Sum a float32 grid across MPI ranks in chunks. Rank 0 receives in-place."""
    if grid_local.dtype != np.float32:
        raise TypeError(f"{label} reduction expected float32, got {grid_local.dtype}")
    if not grid_local.flags.c_contiguous:
        grid_local = np.ascontiguousarray(grid_local)
    rank = comm.Get_rank()
    chunk_planes = _socket_reduce_chunk_planes(grid_local.shape)
    n_chunks = (grid_local.shape[0] + chunk_planes - 1) // chunk_planes
    t0 = time.perf_counter()
    log(
        f"{label}: MPI sum-reduce start, grid={grid_local.shape}, "
        f"chunk_planes={chunk_planes}, n_chunks={n_chunks}, RSS={_rss_gb():.1f} GB"
    )
    for chunk_idx, i0 in enumerate(range(0, grid_local.shape[0], chunk_planes)):
        i1 = min(grid_local.shape[0], i0 + chunk_planes)
        chunk = grid_local[i0:i1]
        if rank == 0:
            comm.Reduce(MPI.IN_PLACE, chunk, op=MPI.SUM, root=0)
            if chunk_idx == 0 or chunk_idx == n_chunks - 1 or (chunk_idx + 1) % 8 == 0:
                log(
                    f"{label}: MPI chunk {chunk_idx + 1}/{n_chunks} reduced "
                    f"({i0}:{i1})",
                    leader_only=True,
                )
        else:
            comm.Reduce(chunk, None, op=MPI.SUM, root=0)
            if chunk_idx == 0 or chunk_idx == n_chunks - 1 or (chunk_idx + 1) % 8 == 0:
                log(
                    f"{label}: MPI chunk {chunk_idx + 1}/{n_chunks} sent "
                    f"({i0}:{i1})"
                )
    if rank == 0:
        log(f"{label}: MPI sum-reduce done in {time.perf_counter() - t0:.1f} s")
        return grid_local
    log(f"{label}: MPI sum-reduce send done in {time.perf_counter() - t0:.1f} s")
    return None


def _sum_grid_to_root_inplace(grid_local: np.ndarray, backend: str, comm, label: str):
    if grid_local.dtype != np.float32:
        raise TypeError(f"{label} reduction expected float32, got {grid_local.dtype}")
    if not grid_local.flags.c_contiguous:
        grid_local = np.ascontiguousarray(grid_local)
    if process_count <= 1:
        return grid_local
    if backend == "socket":
        return _socket_sum_grid_to_root_inplace(grid_local, label)
    if backend == "mpi":
        return _mpi_sum_grid_to_root_inplace(grid_local, comm, label)
    raise ValueError(f"Unsupported CPU grid reduction backend {backend!r}")


def _mpi_allreduce_int(value: int, comm) -> int:
    if comm is None:
        gathered = multihost_utils.process_allgather(np.asarray(value, dtype=np.int64), tiled=False)
        return int(np.asarray(gathered, dtype=np.int64).sum())
    return int(comm.allreduce(int(value), op=MPI.SUM))


def _allreduce_scalar(value: float, op: str, comm) -> float:
    if comm is not None:
        mpi_op = MPI.MIN if op == "min" else MPI.MAX
        return float(comm.allreduce(float(value), op=mpi_op))
    gathered = np.asarray(
        multihost_utils.process_allgather(np.asarray(value, dtype=np.float32), tiled=False),
        dtype=np.float32,
    )
    return float(np.min(gathered) if op == "min" else np.max(gathered))


def _get_postprocess_comm():
    """Deprecated compatibility wrapper for older local experiments."""
    backend, comm = _get_postprocess_backend()
    if backend == "serial":
        return None
    if backend != "mpi":
        if process_count > 1:
            raise RuntimeError(
                "_get_postprocess_comm only supports the MPI backend; use "
                "_get_postprocess_backend for socket reductions."
            )
        return None
    return comm


def _local_particles_from_sharded_array(arr, label: str) -> np.ndarray:
    """Move only this process's addressable shards to host as an (N,3) array."""
    shards = getattr(arr, "addressable_shards", None)
    if shards is None:
        host = np.asarray(arr)
        return np.array(host.reshape(-1, host.shape[-1]), dtype=np.float32, copy=True, order="C")
    if len(shards) == 0:
        return np.empty((0, int(arr.shape[-1])), dtype=np.float32)

    local_parts = []
    for shard in shards:
        shard_host = np.asarray(shard.data)
        local_parts.append(shard_host.reshape(-1, shard_host.shape[-1]))

    if len(local_parts) == 1:
        out = np.array(local_parts[0], dtype=np.float32, copy=True, order="C")
    else:
        out = np.ascontiguousarray(np.concatenate(local_parts, axis=0).astype(np.float32, copy=False))
    log(
        f"{label}: copied {out.shape[0]:,} local particles from "
        f"{len(local_parts)} addressable shard(s), RSS={_rss_gb():.1f} GB"
    )
    return out


def _empty_random_selection(grid: int):
    return np.arange(grid**3), 0, 0


def _select_subboxes(dmo_fields_all_rs: np.ndarray, get_randsel: bool,
                     grid: int, nrand_sel_box: int):
    if (nrand_sel_box < grid**3) and get_randsel:
        rng       = np.random.default_rng(0)
        Npart_sum = dmo_fields_all_rs[..., 0].sum(axis=(1, 2, 3))

        npart_min, npart_max = np.percentile(Npart_sum, [2.0, 98.0])
        Npart_clipped = np.clip(Npart_sum, npart_min, npart_max)

        hist, bins_edges = np.histogram(Npart_clipped, bins=16)
        bins_edges[0]  = 0.0
        bins_edges[-1] = bins_edges[-1] * 100
        nsel_per_jb    = nrand_sel_box // len(hist)

        indsel_all = []
        for jbd in range(len(bins_edges) - 1):
            indsel = np.where((Npart_sum >= bins_edges[jbd]) &
                              (Npart_sum <  bins_edges[jbd + 1]))[0]
            n_sel  = min(len(indsel), nsel_per_jb)
            if n_sel > 0:
                indsel_all.append(rng.choice(indsel, n_sel, replace=False))

        indsel_all = np.concatenate(indsel_all)
        if len(indsel_all) < nrand_sel_box:
            remaining  = np.setdiff1d(np.arange(grid**3), indsel_all)
            indsel_all = np.concatenate([
                indsel_all,
                rng.choice(remaining, nrand_sel_box - len(indsel_all), replace=False)
            ])
        rand_sel      = rng.permutation(indsel_all)
        Npart_sum_sel = Npart_sum[rand_sel]
        return rand_sel, Npart_sum, Npart_sum_sel
    return _empty_random_selection(grid)


def _write_rs_channel(out: np.ndarray, channel: int, mat_rs: np.ndarray,
                      *, scale: float | None = None, log1p: bool = False) -> None:
    view = mat_rs.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3])
    if log1p:
        np.log1p(view, out=out[..., channel])
    elif scale is None:
        out[..., channel] = view
    else:
        np.multiply(view, np.float32(scale), out=out[..., channel])


def _init_fields_from_global_density(Npart: np.ndarray, get_env: bool, get_vel: bool,
                                     grid: int, grid_sbox: int,
                                     N_bar_vox: float, norm_delta: float):
    n_channels = 2 + (3 if get_env else 0) + (3 if get_vel else 0)
    out = np.empty((grid**3, grid_sbox, grid_sbox, grid_sbox, n_channels), dtype=np.float32)
    inv_norm_delta = 1.0 / (N_bar_vox * norm_delta)

    t0 = time.perf_counter()
    log(f"Root field assembly: base density reshape start, RSS={_rss_gb():.1f} GB", leader_only=True)
    Npart_rs = mat_reshape_fast(Npart, grid, grid_sbox)
    _write_rs_channel(out, 0, Npart_rs, scale=inv_norm_delta)
    _write_rs_channel(out, 1, Npart_rs, log1p=True)
    next_channel = 2
    log(f"Root field assembly: base density channels done in {time.perf_counter() - t0:.1f} s", leader_only=True)

    if get_env:
        t0 = time.perf_counter()
        log("Root field assembly: env pad1 start", leader_only=True)
        Npart_pad1_rs, _ = get_padded_mat_numba(Npart, grid_sbox, grid_sbox, grid)
        _write_rs_channel(out, next_channel, Npart_pad1_rs, scale=inv_norm_delta)
        _write_rs_channel(out, next_channel + 1, Npart_pad1_rs, log1p=True)
        next_channel += 2
        del Npart_pad1_rs
        gc.collect()
        log(f"Root field assembly: env pad1 done in {time.perf_counter() - t0:.1f} s", leader_only=True)

        t0 = time.perf_counter()
        log("Root field assembly: env pad2 start", leader_only=True)
        Npart_pad2_rs, _ = get_padded_mat_numba(Npart, 2 * grid_sbox, grid_sbox, grid)
        _write_rs_channel(out, next_channel, Npart_pad2_rs, scale=inv_norm_delta)
        next_channel += 1
        del Npart_pad2_rs
        gc.collect()
        log(f"Root field assembly: env pad2 done in {time.perf_counter() - t0:.1f} s", leader_only=True)

    return out, next_channel


# =============================================================================
# FIELD PROCESSING  (verbatim from pm.py)
# =============================================================================
def process_LH_sim_fast(pos_m_truth, vel_m_truth, get_env=False, get_vel=False,
                        get_randsel=False, grid=64, grid_sbox=8, nrand_sel_box=32768):
    norm_delta = 10
    norm_vel   = 100
    BoxSize    = boxsize
    MAS_type   = 'CIC'
    grid_tot   = grid_sbox * grid

    rho_bar    = len(pos_m_truth) / BoxSize**3
    vol_vox    = (BoxSize / grid_tot)**3
    N_bar_vox  = rho_bar * vol_vox

    Npart = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
    pos_copy = np.array(pos_m_truth, dtype=np.float32, copy=True)
    MASL.MA(pos_copy, Npart, BoxSize, MAS_type, verbose=False)
    Npart_rs = mat_reshape_fast(Npart, grid, grid_sbox)

    fields_list = [Npart_rs[..., None] / (N_bar_vox * norm_delta),
                   np.log1p(Npart_rs)[..., None]]

    if get_env or get_vel:
        # MAS_type is CIC above, so this is exactly the same density grid.  The
        # old code repainted all particles here, which changed only runtime.
        Npart_cic = Npart

    if get_env:
        Npart_pad1_rs, _ = get_padded_mat_numba(Npart, grid_sbox,     grid_sbox, grid)
        Npart_pad2_rs, _ = get_padded_mat_numba(Npart, 2 * grid_sbox, grid_sbox, grid)
        fields_list.extend([
            Npart_pad1_rs[..., None] / (N_bar_vox * norm_delta),
            np.log1p(Npart_pad1_rs)[..., None],
            Npart_pad2_rs[..., None] / (N_bar_vox * norm_delta),
        ])

    if get_vel:
        vel_m_part = np.zeros((grid_tot, grid_tot, grid_tot, 3), dtype=np.float32)
        vel_copy   = np.array(vel_m_truth, dtype=np.float32, copy=True)
        for jc in range(3):
            mom_jc = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
            MASL.MA(pos_copy, mom_jc, BoxSize, 'CIC', verbose=False, W=vel_copy[:, jc])
            vel_m_jc = np.divide(mom_jc, Npart_cic,
                                 out=np.zeros_like(mom_jc), where=Npart_cic != 0)
            vel_m_part[..., jc] = vel_m_jc / norm_vel
        vel_m_part_rs = mat_reshape_fast(vel_m_part, grid, grid_sbox)
        fields_list.append(vel_m_part_rs)

    dmo_fields_all_snap = np.concatenate(fields_list, axis=-1)
    dmo_fields_all_rs   = dmo_fields_all_snap.reshape((grid**3, *dmo_fields_all_snap.shape[3:]))

    if (nrand_sel_box < grid**3) and get_randsel:
        rng       = np.random.default_rng(0)
        Npart_sum = dmo_fields_all_rs[..., 0].sum(axis=(1, 2, 3))

        npart_min, npart_max = np.percentile(Npart_sum, [2.0, 98.0])
        Npart_clipped = np.clip(Npart_sum, npart_min, npart_max)

        hist, bins_edges = np.histogram(Npart_clipped, bins=16)
        bins_edges[0]  = 0.0
        bins_edges[-1] = bins_edges[-1] * 100
        nsel_per_jb    = nrand_sel_box // len(hist)

        indsel_all = []
        for jbd in range(len(bins_edges) - 1):
            indsel = np.where((Npart_sum >= bins_edges[jbd]) &
                              (Npart_sum <  bins_edges[jbd + 1]))[0]
            n_sel  = min(len(indsel), nsel_per_jb)
            if n_sel > 0:
                indsel_all.append(rng.choice(indsel, n_sel, replace=False))

        indsel_all = np.concatenate(indsel_all)
        if len(indsel_all) < nrand_sel_box:
            remaining  = np.setdiff1d(np.arange(grid**3), indsel_all)
            indsel_all = np.concatenate([
                indsel_all,
                rng.choice(remaining, nrand_sel_box - len(indsel_all), replace=False)
            ])
        rand_sel      = rng.permutation(indsel_all)
        Npart_sum_sel = Npart_sum[rand_sel]
    else:
        rand_sel = np.arange(grid**3)
        Npart_sum = Npart_sum_sel = 0

    dmo_fields_all_rs[~np.isfinite(dmo_fields_all_rs)] = 0.0
    return dmo_fields_all_rs, rand_sel, Npart_sum, Npart_sum_sel, norm_delta, norm_vel


def process_LH_sim_distributed_from_jax(X_g, v_g, get_env=False, get_vel=False,
                                        get_randsel=False, grid=64, grid_sbox=8,
                                        nrand_sel_box=32768):
    """Distributed CPU postprocess preserving the serial CIC physics.

    Each JAX process paints only its local particle shard into a 512^3 CIC
    grid.  The grids are summed across processes, then rank 0 builds exactly
    the same subbox-field tensor and metadata as process_LH_sim_fast.
    """
    backend, comm = _get_postprocess_backend()
    if backend == "serial":
        log("Serial CPU postprocess fallback: all-gather positions start")
        X_sim = np.asarray(all_gather(X_g)).reshape(-1, 3).astype(np.float32)
        if get_vel:
            log("Serial CPU postprocess fallback: all-gather velocities start")
            P_sim = np.asarray(all_gather(v_g)).reshape(-1, 3).astype(np.float32)
        else:
            P_sim = np.empty((0, 3), dtype=np.float32)
        if is_leader:
            out = process_LH_sim_fast(
                X_sim, P_sim, get_env=get_env, get_vel=get_vel,
                get_randsel=get_randsel, grid=grid, grid_sbox=grid_sbox,
                nrand_sel_box=nrand_sel_box,
            )
            del X_sim, P_sim
            gc.collect()
            return out
        del X_sim, P_sim
        gc.collect()
        return None, None, None, None, None, None

    norm_delta = 10
    norm_vel   = 100
    BoxSize    = boxsize
    grid_tot   = grid_sbox * grid
    global_n_particles = int(np.prod(tuple(int(s) for s in X_g.shape[:-1])))
    rho_bar    = global_n_particles / BoxSize**3
    vol_vox    = (BoxSize / grid_tot)**3
    N_bar_vox  = rho_bar * vol_vox

    t_all = time.perf_counter()
    log(
        f"Distributed CPU postprocess start "
        f"(get_env={get_env}, get_vel={get_vel}, get_randsel={get_randsel}, "
        f"global_n_particles={global_n_particles:,}, reduce_backend={backend})"
    )

    t0 = time.perf_counter()
    pos_local = _local_particles_from_sharded_array(X_g, "positions")
    global_count = _mpi_allreduce_int(int(pos_local.shape[0]), comm)
    if global_count != global_n_particles:
        raise RuntimeError(
            f"Distributed postprocess particle-count mismatch: "
            f"sum(local)={global_count:,}, X_g global shape implies "
            f"{global_n_particles:,}."
        )
    log(
        f"Local position copy done in {time.perf_counter() - t0:.1f} s; "
        f"global X=[{_allreduce_scalar(float(pos_local.min()) if pos_local.size else np.inf, 'min', comm):.3g},"
        f"{_allreduce_scalar(float(pos_local.max()) if pos_local.size else -np.inf, 'max', comm):.3g}]"
    )

    t0 = time.perf_counter()
    Npart_local = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
    MASL.MA(pos_local, Npart_local, BoxSize, "CIC", verbose=False)
    log(f"Local density CIC paint done in {time.perf_counter() - t0:.1f} s, RSS={_rss_gb():.1f} GB")

    Npart_global = _sum_grid_to_root_inplace(Npart_local, backend, comm, "density")
    if not is_leader:
        del Npart_local
        if not get_vel:
            del pos_local
        gc.collect()
    else:
        Npart_sum_total = float(Npart_global.sum(dtype=np.float64))
        if abs(Npart_sum_total - global_n_particles) > max(1e-3, 1e-5 * global_n_particles):
            raise RuntimeError(
                "Global CIC density does not conserve particle count: "
                f"sum={Npart_sum_total:.6g}, expected={global_n_particles}."
            )
        log(
            f"Global density ready on root: sum={Npart_sum_total:.6g}, "
            f"RSS={_rss_gb():.1f} GB",
            leader_only=True,
        )

    if is_leader:
        dmo_fields_all_rs, next_channel = _init_fields_from_global_density(
            Npart_global, get_env, get_vel, grid, grid_sbox, N_bar_vox, norm_delta,
        )
    else:
        dmo_fields_all_rs = None
        next_channel = None

    if get_vel:
        t0 = time.perf_counter()
        vel_local = _local_particles_from_sharded_array(v_g, "velocities")
        log(
            f"Local velocity copy done in {time.perf_counter() - t0:.1f} s; "
            f"global v=[{_allreduce_scalar(float(vel_local.min()) if vel_local.size else np.inf, 'min', comm):.3g},"
            f"{_allreduce_scalar(float(vel_local.max()) if vel_local.size else -np.inf, 'max', comm):.3g}] km/s"
        )
        if vel_local.shape[0] != pos_local.shape[0]:
            raise RuntimeError(
                f"Local position/velocity count mismatch: "
                f"{pos_local.shape[0]} vs {vel_local.shape[0]}"
            )

        for jc in range(3):
            t0 = time.perf_counter()
            mom_jc = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
            MASL.MA(pos_local, mom_jc, BoxSize, "CIC", verbose=False, W=vel_local[:, jc])
            log(
                f"Local momentum CIC paint component {jc} done in "
                f"{time.perf_counter() - t0:.1f} s"
            )
            mom_global = _sum_grid_to_root_inplace(mom_jc, backend, comm, f"momentum[{jc}]")
            if is_leader:
                t0 = time.perf_counter()
                np.divide(
                    mom_global,
                    Npart_global,
                    out=mom_global,
                    where=Npart_global != 0,
                )
                mom_global[Npart_global == 0] = 0.0
                mom_global /= np.float32(norm_vel)
                vel_rs = mat_reshape_fast(mom_global, grid, grid_sbox)
                _write_rs_channel(dmo_fields_all_rs, next_channel + jc, vel_rs)
                log(
                    f"Root velocity channel {jc} finalized in "
                    f"{time.perf_counter() - t0:.1f} s",
                    leader_only=True,
                )
                del mom_global
            del mom_jc
            gc.collect()
        del vel_local
        del pos_local
    elif is_leader:
        del pos_local

    if is_leader:
        dmo_fields_all_rs[~np.isfinite(dmo_fields_all_rs)] = 0.0
        rand_sel, Npart_sum, Npart_sum_sel = _select_subboxes(
            dmo_fields_all_rs, get_randsel, grid, nrand_sel_box,
        )
        log(
            f"Distributed CPU postprocess done in {time.perf_counter() - t_all:.1f} s, "
            f"output shape={dmo_fields_all_rs.shape}, RSS={_rss_gb():.1f} GB",
            leader_only=True,
        )
        del Npart_global
        gc.collect()
        return dmo_fields_all_rs, rand_sel, Npart_sum, Npart_sum_sel, norm_delta, norm_vel

    log(f"Distributed CPU postprocess non-root done in {time.perf_counter() - t_all:.1f} s")
    return None, None, None, None, norm_delta, norm_vel


def process_LH_sim_sampled_from_jax(X_g, v_g, subvol_sel: np.ndarray,
                                    get_env=False, get_vel=False,
                                    grid=64, grid_sbox=8):
    """Paint production CIC grids, but assemble only selected subvolumes."""
    backend, comm = _get_postprocess_backend()
    if backend == "serial":
        log("Sampled serial CPU postprocess fallback: all-gather positions start")
        X_sim = np.asarray(all_gather(X_g)).reshape(-1, 3).astype(np.float32)
        if get_vel:
            P_sim = np.asarray(all_gather(v_g)).reshape(-1, 3).astype(np.float32)
        else:
            P_sim = np.empty((0, 3), dtype=np.float32)
        if is_leader:
            full, _, _, _, norm_delta, norm_vel = process_LH_sim_fast(
                X_sim, P_sim, get_env=get_env, get_vel=get_vel,
                get_randsel=False, grid=grid, grid_sbox=grid_sbox,
                nrand_sel_box=len(subvol_sel),
            )
            sampled = full[subvol_sel]
            del full, X_sim, P_sim
            gc.collect()
            return sampled, subvol_sel, None, None, norm_delta, norm_vel
        del X_sim, P_sim
        gc.collect()
        return None, None, None, None, None, None

    norm_delta = 10
    norm_vel = 100
    BoxSize = boxsize
    grid_tot = grid_sbox * grid
    subvol_sel = np.asarray(subvol_sel, dtype=np.int64)
    global_n_particles = int(np.prod(tuple(int(s) for s in X_g.shape[:-1])))
    rho_bar = global_n_particles / BoxSize**3
    vol_vox = (BoxSize / grid_tot)**3
    N_bar_vox = rho_bar * vol_vox
    inv_norm_delta = np.float32(1.0 / (N_bar_vox * norm_delta))

    t_all = time.perf_counter()
    log(
        f"Sampled CPU postprocess start "
        f"(get_env={get_env}, get_vel={get_vel}, n_subvols={len(subvol_sel):,}, "
        f"global_n_particles={global_n_particles:,}, reduce_backend={backend})"
    )

    pos_local = _local_particles_from_sharded_array(X_g, "positions")
    global_count = _mpi_allreduce_int(int(pos_local.shape[0]), comm)
    if global_count != global_n_particles:
        raise RuntimeError(
            f"Sampled postprocess particle-count mismatch: "
            f"sum(local)={global_count:,}, X_g global shape implies "
            f"{global_n_particles:,}."
        )

    t0 = time.perf_counter()
    Npart_local = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
    MASL.MA(pos_local, Npart_local, BoxSize, "CIC", verbose=False)
    log(f"Sampled local density CIC paint done in {time.perf_counter() - t0:.1f} s, RSS={_rss_gb():.1f} GB")

    Npart_global = _sum_grid_to_root_inplace(Npart_local, backend, comm, "sampled_density")
    if not is_leader:
        del Npart_local
        if not get_vel:
            del pos_local
        gc.collect()
        sampled_fields = None
        Npart_sum_sel = None
    else:
        Npart_sum_total = float(Npart_global.sum(dtype=np.float64))
        if abs(Npart_sum_total - global_n_particles) > max(1e-3, 1e-5 * global_n_particles):
            raise RuntimeError(
                "Sampled global CIC density does not conserve particle count: "
                f"sum={Npart_sum_total:.6g}, expected={global_n_particles}."
            )
        env_channels = 3 if get_env else 0
        vel_offset = 2 + env_channels
        n_channels = vel_offset + (3 if get_vel else 0)
        sampled_fields = np.empty(
            (len(subvol_sel), grid_sbox, grid_sbox, grid_sbox, n_channels),
            dtype=np.float32,
        )
        density_blocks = _extract_subvolumes_from_grid(Npart_global, subvol_sel, grid, grid_sbox)
        np.multiply(density_blocks, inv_norm_delta, out=sampled_fields[..., 0])
        np.log1p(density_blocks, out=sampled_fields[..., 1])
        Npart_sum_sel = sampled_fields[..., 0].sum(axis=(1, 2, 3))
        log(
            f"Sampled density ready on root: sum={Npart_sum_total:.6g}, "
            f"sample_shape={sampled_fields.shape}, RSS={_rss_gb():.1f} GB",
            leader_only=True,
        )
        del density_blocks
        gc.collect()
        if get_env:
            t0 = time.perf_counter()
            pad1_blocks = extract_padded_subvolumes_numba(
                Npart_global, subvol_sel, grid_sbox, grid_sbox, grid
            )
            np.multiply(pad1_blocks, inv_norm_delta, out=sampled_fields[..., 2])
            np.log1p(pad1_blocks, out=sampled_fields[..., 3])
            del pad1_blocks
            gc.collect()
            log(
                f"Sampled env pad1 ready in {time.perf_counter() - t0:.1f} s, "
                f"RSS={_rss_gb():.1f} GB",
                leader_only=True,
            )

            t0 = time.perf_counter()
            pad2_blocks = extract_padded_subvolumes_numba(
                Npart_global, subvol_sel, 2 * grid_sbox, grid_sbox, grid
            )
            np.multiply(pad2_blocks, inv_norm_delta, out=sampled_fields[..., 4])
            del pad2_blocks
            gc.collect()
            log(
                f"Sampled env pad2 ready in {time.perf_counter() - t0:.1f} s, "
                f"RSS={_rss_gb():.1f} GB",
                leader_only=True,
            )

    if get_vel:
        vel_local = _local_particles_from_sharded_array(v_g, "velocities")
        if vel_local.shape[0] != pos_local.shape[0]:
            raise RuntimeError(
                f"Local position/velocity count mismatch: "
                f"{pos_local.shape[0]} vs {vel_local.shape[0]}"
            )
        for jc in range(3):
            t0 = time.perf_counter()
            mom_jc = np.zeros((grid_tot, grid_tot, grid_tot), dtype=np.float32)
            MASL.MA(pos_local, mom_jc, BoxSize, "CIC", verbose=False, W=vel_local[:, jc])
            log(
                f"Sampled local momentum CIC paint component {jc} done in "
                f"{time.perf_counter() - t0:.1f} s"
            )
            mom_global = _sum_grid_to_root_inplace(mom_jc, backend, comm, f"sampled_momentum[{jc}]")
            if is_leader:
                np.divide(
                    mom_global,
                    Npart_global,
                    out=mom_global,
                    where=Npart_global != 0,
                )
                mom_global[Npart_global == 0] = 0.0
                mom_global /= np.float32(norm_vel)
                _extract_subvolumes_from_grid(
                    mom_global, subvol_sel, grid, grid_sbox,
                    out=sampled_fields[..., vel_offset + jc],
                )
                log(f"Sampled velocity channel {jc} ready", leader_only=True)
                del mom_global
            del mom_jc
            gc.collect()
        del vel_local
        del pos_local
    elif is_leader:
        del pos_local

    if is_leader:
        sampled_fields[~np.isfinite(sampled_fields)] = 0.0
        del Npart_global
        gc.collect()
        log(
            f"Sampled CPU postprocess done in {time.perf_counter() - t_all:.1f} s, "
            f"output shape={sampled_fields.shape}, RSS={_rss_gb():.1f} GB",
            leader_only=True,
        )
        return sampled_fields, subvol_sel, None, Npart_sum_sel, norm_delta, norm_vel

    log(f"Sampled CPU postprocess non-root done in {time.perf_counter() - t_all:.1f} s")
    return None, None, None, None, norm_delta, norm_vel


# =============================================================================
# NUMBA WARMUP
# =============================================================================
def warmup_numba():
    log("Warming up numba JIT", leader_only=True)
    dummy     = np.random.randn(64, 64, 64).astype(np.float32)
    dummy_pad = np.pad(dummy, 4, mode='wrap').astype(np.float32)
    _get_padded_mat_numba_core(dummy_pad, 4, 8, 8, 3)
    log("Numba warmup done", leader_only=True)


# =============================================================================
# MAIN
# =============================================================================
def run_one_simulation_multigpu(sim_id, savefull=False):
    os.makedirs(root_out, exist_ok=True)
    if (diagnostic_sampled or diagnostic_state_stats or diagnostic_initial_state_stats) and savefull:
        raise ValueError(
            "DISCO_DIAGNOSTIC_MODE outputs require savefull=0. "
            "Pass 0 as the second command-line argument."
        )

    savefname  = root_out + 'dmo_fields_subvols_grid_%d_CV_%d_%s.npy'  % (grid_sbox, sim_id, output_tag)
    meta_fname = root_out + 'meta_dmo_fields_subvols_grid_%d_CV_%d_%s.pkl' % (grid_sbox, sim_id, output_tag)
    ic_override = os.environ.get("DISCO_IC_PATH", "").strip()
    ic_template = os.environ.get("DISCO_IC_PATH_TEMPLATE", "").strip()
    if ic_override:
        ic_fname = ic_override
    elif ic_template:
        ic_fname = ic_template.format(sim_id=sim_id)
    else:
        ic_fname = path_ic + "IC_CV%d_3gpc.npy" % sim_id

    if os.path.exists(savefname) and os.path.exists(meta_fname):
        log(f"Outputs already exist for sim_id={sim_id}, skipping.")
        return
    
    log(
        f"Running simulation {sim_id} on {n_dev} global devices "
        f"({local_n_dev} local) with lpt_pdims={pdims}, pm_pdims={pm_pdims}"
    )
    log(
        "Cosmology: "
        f"Omega_m={Om:.5f}, Omega_b={Ob:.5f}, h={h:.5f}, "
        f"n_s={ns:.5f}, sigma8={sigma8:.5f} "
        f"(source={cosmo_source})",
        leader_only=True,
    )
    mem_mon = MemoryMonitor(interval=1.0).start()
    _t0_disco = time.perf_counter()
    t_disco = 0.0   # accumulates: IC load + setup + DKD + gather (excludes field processing)

    # ------------------------------------------------------------------
    # 1. Load IC delta from disk (host).
    # ------------------------------------------------------------------
    _t0_load = time.perf_counter()
    log(f"Loading IC for sim_id={sim_id}: {ic_fname}")
    ic_np = np.load(ic_fname).astype(np.float32, copy=False)
    log(
        f"IC loaded: shape={ic_np.shape}, dtype={ic_np.dtype}, "
        f"elapsed={time.perf_counter() - _t0_load:.1f} s"
    )

    # ------------------------------------------------------------------
    # 2. Distributed nLPT initial state via discodj_dist.
    #    - Under multi-host JAX, do not force DiscoDJ(device="gpu");
    #      distributed sharding controls device placement.
    #    - We compute fphi_full ourselves under jax.jit (see
    #      _compute_fphi_full_sharded) so the inverse-Laplace kernel is
    #      built sharded; otherwise with_external_ics(delta=...) OOMs at
    #      res=1536 on the leader GPU.  Inject via update(ics=...).
    #    - with_lpt(n_order, sharding=sharding_disp) is internally jit'd by
    #      discodj_dist, so its inv_laplace stays sharded under XLA.
    #    - evaluate_lpt_psi_at_a / _evaluate_lpt_property_at_a return the
    #      LPT displacement and its dD-derivative (= integrator momentum
    #      when time_var='D'), already sharded as P('x','y',None,None).
    # ------------------------------------------------------------------
    dj_dist = DiscoDJ(dim=dim, res=res, device=discodj_device, precision=precision,
                      boxsize=boxsize, cosmo=cosmo, name=f'sim_{sim_id}_dist')
    log("Building DiscoDJ growth timetables")
    dj_dist = dj_dist.with_timetables()
    Dplus_aic = float(np.interp(jnp.log10(a_ic),
                                jnp.log10(dj_dist.cosmo._timetables['a']),
                                dj_dist.cosmo._timetables['Dplus']))
    log(f"D+(a_ic={a_ic:.5f}) = {Dplus_aic:.4e}")

    # Rescale delta from a=a_ic to a=1 (linear theory) before distributed FFT.
    log("Rescaling IC delta from a_ic to z=0 linear amplitude")
    ic_np /= Dplus_aic   # in-place: avoid 14.5 GB extra host buffer at res=1536
    log("IC rescale complete")

    log("Distributed IC FFT + nLPT starting")
    _t0_lpt = time.perf_counter()
    log("Creating sharded JAX IC delta from local host slices")
    delta_g = _host_array_to_global_sharded(ic_np, sharding_field, dtype=dtype)
    del ic_np
    gc.collect()
    log("Computing sharded fphi_full from IC delta")
    fphi_full_g = _compute_fphi_full_sharded(delta_g)
    fphi_full_g.block_until_ready()
    del delta_g
    gc.collect()
    jax.clear_caches()
    log(f"fphi_full ready: sharding={fphi_full_g.sharding}")
    initial_spectral_psi1_rms = None
    initial_direct_psi1_component_stats = None
    if diagnostic_initial_state_stats:
        log("Collecting initial fphi spectral psi1 RMS")
        initial_spectral_psi1_rms = _collect_spectral_psi1_rms(fphi_full_g)
        log("Collecting initial direct-iFFT psi1 stats")
        initial_direct_psi1_component_stats = _collect_direct_psi1_stats(fphi_full_g)
        if _get_env_bool("DISCO_INITIAL_STATE_SKIP_2LPT", False):
            t_disco = time.perf_counter() - _t0_disco
            if is_leader:
                log(
                    f"Saving initial direct-psi1 diagnostic array to {savefname}",
                    leader_only=True,
                )
                np.save(
                    savefname,
                    initial_direct_psi1_component_stats[None, None, ...].astype(np.float32),
                )
                saved = {
                    'cosmo':           cosmo,
                    'cosmo_source':    cosmo_source,
                    'zsnaps':          z_snaps,
                    'boxsize':         boxsize,
                    'res':             res,
                    'factor':          factor,
                    'res_pm':          res_pm,
                    'ic_fname':        ic_fname,
                    'lpt_order':       n_order,
                    'lpt_grad_kernel_order': lpt_grad_kernel_order,
                    'pm_grad_kernel_order':  grad_kernel_order,
                    'diagnostic_max_pm_steps': diagnostic_max_pm_steps,
                    'output_tag':       output_tag,
                    'diagnostic_mode':  diagnostic_mode,
                    'diagnostic_initial_state_stats': True,
                    'initial_state_skipped_2lpt': True,
                    'initial_spectral_psi1_rms': initial_spectral_psi1_rms,
                    'initial_direct_psi1_component_stats': initial_direct_psi1_component_stats,
                    'ic_fft_backend':   ic_fft_backend,
                    'lpt_fft_backend':  lpt_fft_backend,
                    'lpt_mu2_fft_backend': lpt_mu2_fft_backend,
                    'lpt_pdims':        pdims,
                    'pm_pdims':         pm_pdims,
                    't_disco_s':        t_disco,
                }
                pk.dump(saved, open(meta_fname, 'wb'))
                log(f"Saved metadata to {meta_fname}", leader_only=True)
                log(f"Saved {savefname}", leader_only=True)
            mem_mon.stop()
            log("Memory summary")
            mem_mon.log_summary()
            del fphi_full_g, dj_dist
            gc.collect()
            jax.clear_caches()
            return
    if n_order == 1:
        log("Using fast distributed 1LPT initializer")
        psi_g, mom_g = _compute_1lpt_initial_state_from_fphi(fphi_full_g, Dplus_aic)
        del fphi_full_g
        gc.collect()
        jax.clear_caches()
        log(f"Distributed IC FFT + fast 1LPT done: {time.perf_counter()-_t0_lpt:.1f} s")
    elif n_order == 2:
        log("Using memory-optimized distributed 2LPT initializer")
        psi_g, mom_g = compute_2lpt_initial_state_distributed(
            fphi_full_g,
            res=res,
            boxsize=boxsize,
            Dplus=Dplus_aic,
            grad_kernel_order=lpt_grad_kernel_order,
            dtype_num=dtype_num,
            dtype_c_num=dtype_c_num,
            no_factors=False,
            field_sharding=sharding_field,
            disp_sharding=sharding_disp,
            fft_sharding=getattr(fphi_full_g, "sharding", None),
            fft_backend=lpt_fft_backend,
            mu2_fft_backend=lpt_mu2_fft_backend,
            progress=_get_env_bool("DISCO_PROGRESS_2LPT", True),
            progress_prefix=f"[proc {process_id}/{process_count}]",
            sync_progress=_get_env_bool("DISCO_PROGRESS_SYNC", True),
        )
        del fphi_full_g
        gc.collect()
        jax.clear_caches()
        psi_g.block_until_ready()
        mom_g.block_until_ready()
        log(f"Distributed IC FFT + optimized 2LPT done: {time.perf_counter()-_t0_lpt:.1f} s")
    else:
        log(f"Using generic distributed {n_order}LPT path")
        dj_dist = dj_dist.update(ics={"fphi_full": fphi_full_g})
        dj_dist = dj_dist.with_lpt(n_order=n_order, grad_kernel_order=lpt_grad_kernel_order,
                                   sharding=sharding_disp, try_to_jit=True,
                                   lpt_fft_backend=lpt_fft_backend)
        dj_dist._lpt.psi[f'psi_{n_order}'].block_until_ready()
        log(f"Distributed IC FFT + {n_order}LPT done: {time.perf_counter()-_t0_lpt:.1f} s")
        log(f"psi_{n_order} sharding: {dj_dist._lpt.psi[f'psi_{n_order}'].sharding}")

        # Initial displacement and dPsi/dD (integrator momentum in D-time).
        log("Evaluating LPT displacement and momentum at a_ic")
        psi_g = dj_dist.evaluate_lpt_psi_at_a(a_ic, n_order=n_order).astype(dtype)
        mom_g = dj_dist._evaluate_lpt_property_at_a(
            a=a_ic, n_order=n_order, include_psi_0=False, D_derivative=True,
        ).astype(dtype)
    psi_g = lax.with_sharding_constraint(psi_g, sharding_disp)
    mom_g = lax.with_sharding_constraint(mom_g, sharding_disp)
    psi_g.block_until_ready()
    mom_g.block_until_ready()
    log(f"Initial psi/momentum ready after LPT: psi sharding={psi_g.sharding}")
    if diagnostic_initial_state_stats:
        log("Collecting initial real-space state stats before PM evolution")
        F_ic = float(dj_dist.cosmo.Fplus(a_ic))
        initial_state_stats = _collect_state_stats(
            psi_g, mom_g, snapshot=-1, z=(1.0 / a_ic) - 1.0,
            a_end=a_ic, F_end=F_ic,
        )
        t_disco = time.perf_counter() - _t0_disco
        if is_leader:
            log(f"Saving initial-state diagnostic array to {savefname}", leader_only=True)
            np.save(
                savefname,
                initial_state_stats["component_stats"][None, ...].astype(np.float32),
            )
            saved = {
                'cosmo':           cosmo,
                'cosmo_source':    cosmo_source,
                'zsnaps':          z_snaps,
                'boxsize':         boxsize,
                'res':             res,
                'factor':          factor,
                'res_pm':          res_pm,
                'ic_fname':        ic_fname,
                'lpt_order':       n_order,
                'lpt_grad_kernel_order': lpt_grad_kernel_order,
                'pm_grad_kernel_order':  grad_kernel_order,
                'diagnostic_max_pm_steps': diagnostic_max_pm_steps,
                'output_tag':       output_tag,
                'diagnostic_mode':  diagnostic_mode,
                'diagnostic_initial_state_stats': True,
                'initial_spectral_psi1_rms': initial_spectral_psi1_rms,
                'initial_direct_psi1_component_stats': initial_direct_psi1_component_stats,
                'initial_state_stats': initial_state_stats,
                'ic_fft_backend':   ic_fft_backend,
                'lpt_fft_backend':  lpt_fft_backend,
                'lpt_mu2_fft_backend': lpt_mu2_fft_backend,
                'lpt_pdims':        pdims,
                'pm_pdims':         pm_pdims,
                't_disco_s':        t_disco,
            }
            pk.dump(saved, open(meta_fname, 'wb'))
            log(f"Saved metadata to {meta_fname}", leader_only=True)
            log(f"Saved {savefname}", leader_only=True)
        mem_mon.stop()
        log("Memory summary")
        mem_mon.log_summary()
        del dj_dist, psi_g, mom_g
        gc.collect()
        jax.clear_caches()
        return
    if needs_pm_reshard:
        log(
            f"Resharding initial state from LPT mesh {pdims}/('x','y') "
            f"to PM mesh {pm_pdims}/{pm_axis_names}"
        )
        _t0_reshard = time.perf_counter()
        psi_g, mom_g = _reshard_lpt_state_to_pm_mesh(psi_g, mom_g)
        psi_g.block_until_ready()
        mom_g.block_until_ready()
        gc.collect()
        jax.clear_caches()
        log(f"PM mesh reshard complete in {time.perf_counter() - _t0_reshard:.1f} s")

    cosmo_dj = dj_dist.cosmo
    del dj_dist
    gc.collect()
    t_disco = time.perf_counter() - _t0_disco   # IC load + distributed LPT
    log(f"Initial condition stage complete: cumulative DiscoDJ time={t_disco:.1f} s")

    # ------------------------------------------------------------------
    # 3. DKD kernel and snapshot position/velocity conversion
    # ------------------------------------------------------------------
    @partial(jax.jit, donate_argnums=(0, 1))
    def run_dkd_step_dist(psi, mom, args_step):
        d1, d2, alpha, beta = args_step
        psi_mid = psi + d1 * mom
        max_x = jnp.max(jnp.abs(psi_mid[..., 0] / pm_cell_size))
        max_y = jnp.max(jnp.abs(psi_mid[..., 1] / pm_cell_size))
        max_z = jnp.max(jnp.abs(psi_mid[..., 2] / pm_cell_size))
        max_all = jnp.maximum(jnp.maximum(max_x, max_y), max_z)
        halo_stats = jnp.stack([max_all, max_x, max_y, max_z]).astype(dtype)
        mom_new = kick_PM_distributed(
            psi_mid, mom, alpha=alpha, beta=beta,
            dim=dim, res_pm=res_pm, boxsize=boxsize,
            halo_size=halo_size, sharding=sharding_pm_disp,
            grad_order=grad_kernel_order,
            lap_order=laplace_kernel_order,
            dtype_num=dtype_num, worder=worder,
            deconvolve=deconvolve,
            fft_backend=pm_fft_backend)
        acc_from_kick = (mom_new - alpha * mom) / beta
        acc_rms = jnp.sqrt(jnp.mean(acc_from_kick * acc_from_kick, axis=(0, 1, 2)))
        mom_rms = jnp.sqrt(jnp.mean(mom * mom, axis=(0, 1, 2)))
        cross = jnp.sum(acc_from_kick * mom, axis=(0, 1, 2))
        acc2 = jnp.sum(acc_from_kick * acc_from_kick, axis=(0, 1, 2))
        mom2 = jnp.sum(mom * mom, axis=(0, 1, 2))
        tiny = jnp.asarray(1e-30, dtype=dtype)
        acc_over_mom_slope = cross / jnp.maximum(mom2, tiny)
        acc_mom_corr = cross / jnp.sqrt(jnp.maximum(acc2 * mom2, tiny))
        step_stats = jnp.concatenate([
            acc_rms.astype(dtype),
            mom_rms.astype(dtype),
            acc_over_mom_slope.astype(dtype),
            acc_mom_corr.astype(dtype),
        ])
        psi_new = psi_mid + d2 * mom_new
        return psi_new, mom_new, halo_stats, step_stats

    def run_dkd_steps_dist(psi, mom, args_steps_host):
        n_steps = int(args_steps_host.shape[0])
        for step_idx in range(n_steps):
            log(f"PM step {step_idx + 1}/{n_steps}: force + DKD update start")
            args_step = jnp.asarray(args_steps_host[step_idx], dtype=dtype)
            psi, mom, halo_stats, step_stats = run_dkd_step_dist(psi, mom, args_step)
            psi.block_until_ready()
            halo_stats = np.asarray(halo_stats.block_until_ready(), dtype=np.float64)
            if diagnostic_step_stats:
                step_stats_np = np.asarray(step_stats.block_until_ready(), dtype=np.float64)
                expected_slope = (
                    (1.0 - float(args_steps_host[step_idx, 2])) /
                    float(args_steps_host[step_idx, 3])
                )
                log(
                    f"PM step {step_idx + 1}/{n_steps}: kick-implied acc "
                    f"rms_xyz={step_stats_np[0]:.6g},{step_stats_np[1]:.6g},{step_stats_np[2]:.6g}; "
                    f"mom_pre_rms_xyz={step_stats_np[3]:.6g},{step_stats_np[4]:.6g},{step_stats_np[5]:.6g}; "
                    f"acc/mom slope_xyz={step_stats_np[6]:.6g},{step_stats_np[7]:.6g},{step_stats_np[8]:.6g} "
                    f"(ZA expected {expected_slope:.6g}); "
                    f"corr_xyz={step_stats_np[9]:.6g},{step_stats_np[10]:.6g},{step_stats_np[11]:.6g}"
                )
            max_sharded_disp = 0.0
            if pm_pdims[0] > 1:
                max_sharded_disp = max(max_sharded_disp, float(halo_stats[1]))
            if pm_pdims[1] > 1:
                max_sharded_disp = max(max_sharded_disp, float(halo_stats[2]))
            halo_margin = halo_usable_cells - max_sharded_disp
            log(
                f"PM step {step_idx + 1}/{n_steps}: displacement max "
                f"all/x/y/z={halo_stats[0]:.2f}/{halo_stats[1]:.2f}/"
                f"{halo_stats[2]:.2f}/{halo_stats[3]:.2f} PM cells; "
                f"sharded max={max_sharded_disp:.2f}, "
                f"validated limit={halo_usable_cells:.2f}, margin={halo_margin:.2f}"
            )
            if validate_halo and halo_margin < 0.0:
                raise RuntimeError(
                    "Distributed PM halo is too small for this step: "
                    f"sharded displacement {max_sharded_disp:.3f} PM cells exceeds "
                    f"validated limit {halo_usable_cells:.3f} PM cells "
                    f"(halo_size={halo_size}, worder={worder}, "
                    f"safety={halo_safety_cells}). Increase DISCO_HALO_SIZE."
                )
            log(f"PM step {step_idx + 1}/{n_steps}: force + DKD update done")
        return psi, mom

    @jax.jit
    def to_pos_vel(psi_g, mom_g, q_grid, F_end, a_end):
        X_g = jnp.mod(psi_g + q_grid, boxsize)
        v_g = mom_g * F_end / a_end * 100.0   # km/s, matches pm.py convention
        return X_g, v_g

    @jax.jit
    def to_pos_only(psi_g, q_grid):
        return jnp.mod(psi_g + q_grid, boxsize)

    # ------------------------------------------------------------------
    # 4. Loop over snapshots (mirrors pm.py's ja loop)
    # ------------------------------------------------------------------
    rand_sel = Npart_sum = Npart_sum_sel = None
    norm_delta = norm_vel = None
    dmo_fields_all_rs_all = None  # legacy sampled-output path when savefull=False
    sampled_fields_all = []
    state_stats_all = []
    stats_subvol_sel = _make_subvol_selection(
        stats_nsubvols, stats_index_mode, stats_index_seed
    )
    total_channels = 2 + 8 + 8
    channel_offset = 0
    full_output_mm = None
    if is_leader and savefull:
        full_shape = (grid**3, grid_sbox, grid_sbox, grid_sbox, total_channels)
        log(
            f"Creating full output memmap {savefname} with shape={full_shape}, "
            f"dtype=float16",
            leader_only=True,
        )
        full_output_mm = np.lib.format.open_memmap(
            savefname, mode="w+", dtype=np.float16, shape=full_shape
        )
    if stats_subvol_sel is not None:
        if diagnostic_sampled:
            log(
                f"Diagnostic sampled output configured: {len(stats_subvol_sel):,}/{grid**3:,} "
                f"subvolumes (mode={stats_index_mode}, seed={stats_index_seed}), "
                f"mode={diagnostic_mode}, snapshots={diagnostic_snapshot_indices}",
                leader_only=True,
            )
        elif not diagnostic_state_stats:
            log(
                f"Stats sample configured: {len(stats_subvol_sel):,}/{grid**3:,} "
                f"subvolumes (mode={stats_index_mode}, seed={stats_index_seed}); "
                "full simulation output will still be saved.",
                leader_only=True,
            )
    if diagnostic_state_stats:
        log(
            f"State-stats diagnostic configured: snapshots={diagnostic_snapshot_indices}; "
            "host density/velocity painting will be skipped.",
            leader_only=True,
        )

    diagnostic_last_snapshot = (
        max(diagnostic_snapshot_indices)
        if (diagnostic_sampled or diagnostic_state_stats)
        else None
    )
    for ja in range(len(z_snaps)):
        a_ini_ja  = a_init_all[ja]
        a_end_ja  = a_end_all[ja]
        z_ja      = z_snaps[ja]
        nsteps_ja = numsteps_all[ja]
        if ja == 0:
            get_env, get_vel, get_randsel = False, False, False
        elif ja == 1:
            get_env, get_vel, get_randsel = True, True, False
        else:
            get_env, get_vel, get_randsel = True, True, False

        process_snapshot = True
        if diagnostic_sampled or diagnostic_state_stats:
            process_snapshot = ja in diagnostic_snapshot_indices
            get_env = diagnostic_mode == "sampled_density_env"
            get_vel = diagnostic_mode == "sampled_density_velocity"
            get_randsel = False

        if savefull:
            nrand_sel = np.arange(grid**3)
            get_randsel = False

        log(f"Snapshot {ja} start: a={a_ini_ja:.5f}->{a_end_ja:.5f} (z={z_ja}), n_steps={nsteps_ja}")

        solver = DKDPiIntegrator(
            cosmo=cosmo_dj,
            time_dict={'n_steps': nsteps_ja, 'a_ini': a_ini_ja, 'a_end': a_end_ja,
                       'time_var': time_var},
            integrator_name=stepper, use_diffrax=False, dtype_num=dtype_num)
        int_args  = solver.get_integrator_args()
        F_end_ja  = float(cosmo_dj.Fplus(a_end_ja))

        args_dtype_np = np.float64 if dtype_num == 64 else np.float32
        args_steps = np.stack([
            np.asarray(int_args['ddrift1'], dtype=args_dtype_np),
            np.asarray(int_args['ddrift2'], dtype=args_dtype_np),
            np.asarray(int_args['alpha'],   dtype=args_dtype_np),
            np.asarray(int_args['beta'],    dtype=args_dtype_np),
        ], axis=1)  # (nsteps_ja, 4)
        if diagnostic_max_pm_steps is not None:
            requested_steps = int(diagnostic_max_pm_steps)
            if requested_steps < args_steps.shape[0]:
                log(
                    f"Snapshot {ja}: diagnostic truncating PM steps "
                    f"{args_steps.shape[0]} -> {requested_steps}"
                )
                args_steps = args_steps[:requested_steps]

        _t0_snap = time.perf_counter()
        log(f"Snapshot {ja}: DKD PM integration start")
        psi_g, mom_g = run_dkd_steps_dist(psi_g, mom_g, args_steps)
        psi_g.block_until_ready()
        log(f"Snapshot {ja}: DKD PM integration done in {time.perf_counter() - _t0_snap:.1f} s")

        if diagnostic_state_stats:
            if process_snapshot:
                t0_stats = time.perf_counter()
                stats_ja = _collect_state_stats(
                    psi_g, mom_g, snapshot=ja, z=z_ja,
                    a_end=a_end_ja, F_end=F_end_ja,
                )
                if is_leader:
                    state_stats_all.append(stats_ja)
                log(f"Snapshot {ja}: state_stats reduction done in {time.perf_counter() - t0_stats:.1f} s")
            t_disco += time.perf_counter() - _t0_snap
            gc.collect()
            jax.clear_caches()
            multihost_utils.sync_global_devices(f"snap_{ja}_state_stats_done")
            log(f"Snapshot {ja} done")
            if diagnostic_last_snapshot is not None and ja >= diagnostic_last_snapshot:
                log(f"Diagnostic mode={diagnostic_mode}: reached last requested snapshot {ja}; stopping evolution")
                break
            continue

        if not process_snapshot:
            t_disco += time.perf_counter() - _t0_snap
            log(f"Snapshot {ja}: skipping host field processing for diagnostic mode={diagnostic_mode}")
            gc.collect()
            jax.clear_caches()
            multihost_utils.sync_global_devices(f"snap_{ja}_diagnostic_skip_done")
            log(f"Snapshot {ja} done")
            continue

        log(f"Snapshot {ja}: building local q grid and converting to positions/velocities")
        q_local = get_local_q_grid(res_pm=res, boxsize=boxsize,
                                   sharding=sharding_pm_disp, dtype=dtype)
        if get_vel:
            X_g, v_g = to_pos_vel(psi_g, mom_g, q_local, F_end_ja, a_end_ja)
            X_g.block_until_ready()
            v_g.block_until_ready()
        else:
            X_g = to_pos_only(psi_g, q_local)
            X_g.block_until_ready()
            v_g = None
        log(f"Snapshot {ja}: position/velocity conversion ready")
        del q_local
        t_disco += time.perf_counter() - _t0_snap

        # np.save(root_out + "pos_CV%d_z%.1f_3gpc_multigpu.npy" % (sim_id, z_ja), X_sim)
        # np.save(root_out + "vel_CV%d_z%.1f_3gpc_multigpu.npy" % (sim_id, z_ja), P_sim)

        _t0_proc = time.perf_counter()
        log(
            f"Snapshot {ja}: host field processing start "
            f"(get_env={get_env}, get_vel={get_vel}, get_randsel={get_randsel}, "
            f"distributed={distributed_cpu_postprocess})"
        )
        if diagnostic_sampled:
            dmo_fields_all_rs, rand_sel_ja, Npart_sum_ja, Npart_sum_sel_ja, norm_delta_ja, norm_vel_ja = \
                process_LH_sim_sampled_from_jax(
                    X_g, v_g, stats_subvol_sel,
                    get_env=get_env, get_vel=get_vel,
                    grid_sbox=grid_sbox, grid=grid)
        else:
            dmo_fields_all_rs, rand_sel_ja, Npart_sum_ja, Npart_sum_sel_ja, norm_delta_ja, norm_vel_ja = \
                process_LH_sim_distributed_from_jax(
                    X_g, v_g,
                    get_env=get_env, get_vel=get_vel, get_randsel=get_randsel,
                    grid_sbox=grid_sbox, grid=grid, nrand_sel_box=nrand_sel_box)
        t_proc_ja = time.perf_counter() - _t0_proc
        log(f"Snapshot {ja}: host field processing returned in {t_proc_ja:.2f} s")

        del X_g
        if v_g is not None:
            del v_g
        gc.collect()
        jax.clear_caches()

        if is_leader:
            norm_delta = norm_delta_ja
            norm_vel = norm_vel_ja
            if diagnostic_sampled:
                log(
                    f"Snapshot {ja}: sampled diagnostic block ready "
                    f"shape={dmo_fields_all_rs.shape}, RSS={_rss_gb():.1f} GB",
                    leader_only=True,
                )
                sampled_fields_all.append(dmo_fields_all_rs.astype(np.float16))
            elif savefull:
                log(
                    f"Snapshot {ja}: full field block ready for streaming save "
                    f"shape={dmo_fields_all_rs.shape}, RSS={_rss_gb():.1f} GB",
                    leader_only=True,
                )
                _write_snapshot_fields_to_memmap(
                    full_output_mm,
                    dmo_fields_all_rs,
                    channel_offset,
                    output_write_chunk_subvols,
                    f"Snapshot {ja}",
                )
                channel_offset += int(dmo_fields_all_rs.shape[-1])
            else:
                dmo_fields_store = dmo_fields_all_rs
                if ja == 0:
                    dmo_fields_all_rs_all = dmo_fields_store
                else:
                    dmo_fields_all_rs_all = np.concatenate((dmo_fields_all_rs_all, dmo_fields_store), axis=-1)

            if diagnostic_sampled or ja == len(z_snaps) - 1:
                rand_sel      = rand_sel_ja
                Npart_sum     = Npart_sum_ja
                Npart_sum_sel = Npart_sum_sel_ja

            del dmo_fields_all_rs

        gc.collect()
        jax.clear_caches()
        log(f"Snapshot {ja}: waiting for all processes after host processing")
        multihost_utils.sync_global_devices(f"snap_{ja}_host_processing_done")
        log(f"Snapshot {ja} done")
        if diagnostic_last_snapshot is not None and ja >= diagnostic_last_snapshot:
            log(f"Diagnostic mode={diagnostic_mode}: reached last requested snapshot {ja}; stopping evolution")
            break

    # ------------------------------------------------------------------
    # 5. Save concatenated fields (same structure as pm.py)
    # ------------------------------------------------------------------
    if is_leader:
        if diagnostic_state_stats:
            if not state_stats_all:
                raise RuntimeError(
                    f"Diagnostic mode {diagnostic_mode!r} did not process any snapshots."
                )
            log(f"Saving state-stats diagnostic array to {savefname}", leader_only=True)
            fields_to_save = np.stack(
                [item["component_stats"] for item in state_stats_all],
                axis=0,
            ).astype(np.float32)
            np.save(savefname, fields_to_save)
            del fields_to_save
        elif diagnostic_sampled:
            if not sampled_fields_all:
                raise RuntimeError(
                    f"Diagnostic mode {diagnostic_mode!r} did not process any snapshots."
                )
            log(f"Saving sampled diagnostic fields to {savefname}", leader_only=True)
            fields_to_save = np.concatenate(sampled_fields_all, axis=-1)
            np.save(savefname, fields_to_save)
            del fields_to_save
        elif savefull:
            if channel_offset != total_channels:
                raise RuntimeError(
                    f"Full output wrote {channel_offset} channels, expected {total_channels}."
                )
            log(f"Finalizing full streamed field output at {savefname}", leader_only=True)
            full_output_mm.flush()
            del full_output_mm
        else:
            log(f"Saving sampled fields to {savefname}", leader_only=True)
            fields_to_save = dmo_fields_all_rs_all.astype(np.float16)[rand_sel, ...]
            np.save(savefname, fields_to_save)
            del fields_to_save
        saved = {
            'cosmo':           cosmo,
            'cosmo_source':    cosmo_source,
            'zsnaps':          z_snaps,
            'Npart_sum_all':   Npart_sum,
            'Npart_sum_sel':   Npart_sum_sel,
            'norm_delta':      norm_delta,
            'norm_vel':        norm_vel,
            'boxsize':         boxsize,
            'res':             res,
            'grid_sbox':       grid_sbox,
            'grid':            grid,
            'factor':          factor,
            'res_pm':          res_pm,
            'ic_fname':        ic_fname,
            'lpt_order':       n_order,
            'lpt_grad_kernel_order': lpt_grad_kernel_order,
            'pm_grad_kernel_order':  grad_kernel_order,
            'diagnostic_max_pm_steps': diagnostic_max_pm_steps,
            'halo_size':       halo_size,
            'halo_safety_cells': halo_safety_cells,
            'validate_halo':    validate_halo,
            'output_tag':       output_tag,
            'diagnostic_mode':   diagnostic_mode,
            'diagnostic_snapshots': diagnostic_snapshot_indices,
            'diagnostic_sampled_output': bool(diagnostic_sampled),
            'diagnostic_state_stats': bool(diagnostic_state_stats),
            'state_stats':       state_stats_all,
            'saved_full_output': bool(savefull),
            'full_output_shape': (grid**3, grid_sbox, grid_sbox, grid_sbox, total_channels),
            'stats_nsubvols':    stats_nsubvols,
            'stats_index_mode':  stats_index_mode,
            'stats_index_seed':  stats_index_seed,
            'stats_subvol_sel':  stats_subvol_sel,
            'output_write_chunk_subvols': output_write_chunk_subvols,
            'rand_sel':        rand_sel,
            't_disco_s':       t_disco,
        }
        pk.dump(saved, open(meta_fname, 'wb'))
        log(f"Saved metadata to {meta_fname}", leader_only=True)
    mem_mon.stop()
    if is_leader:
        log(f"Saved {savefname}", leader_only=True)
        log(f"Simulation {sim_id} done", leader_only=True)
        log(f"[time] Total DiscoDJ (IC load -> X/P all snaps): {t_disco:.2f} s", leader_only=True)
    log("Memory summary")
    mem_mon.log_summary()

    del cosmo_dj, dmo_fields_all_rs_all
    gc.collect()
    jax.clear_caches()


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    jax.clear_caches()
    gc.collect()
    if _get_env_bool(
        "DISCO_FFT_PREFLIGHT",
        (
            ic_fft_backend.lower() == "cudecomp"
            or lpt_fft_backend.lower() == "cudecomp"
            or pm_fft_backend.lower() == "cudecomp"
            or pm_fft_backend == "JAX_RFFT"
        ),
    ):
        for _fft_backend in sorted({ic_fft_backend, lpt_fft_backend, pm_fft_backend}):
            if _fft_backend == "JAX_RFFT":
                _preflight_pm_rfft_backend()
            else:
                _preflight_fft_backend(_fft_backend)
    if is_leader:
        warmup_numba()
    log("Waiting at numba warmup barrier")
    multihost_utils.sync_global_devices("after_numba_warmup")
    log("Numba warmup barrier passed")

    sim_id = int(sys.argv[1])
    try:
        savefull = bool(int(sys.argv[2]))
    except:
        savefull = True

    run_one_simulation_multigpu(sim_id, savefull=savefull)
    log("Waiting at shutdown barrier")
    multihost_utils.sync_global_devices("before_shutdown")
    log("Shutdown barrier passed")
    if _JAX_DISTRIBUTED:
        log("Shutting down JAX distributed runtime")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
