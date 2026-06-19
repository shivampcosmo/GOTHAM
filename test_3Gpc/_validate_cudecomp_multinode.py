import os
import socket
import sys

import numpy as np

import jax
import jax.numpy as jnp
import jaxdecomp
from jax import lax
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather, sync_global_devices
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

_DISCODJ_SRC = "/mnt/ceph/users/spandey/quijote_v2_gotham/DISCO-DJ/src"
if _DISCODJ_SRC not in sys.path:
    sys.path.insert(0, _DISCODJ_SRC)
from discodj_dist.lpt.nlpt_distributed import (  # noqa: E402
    build_k_vecs_dist,
    compute_2lpt_initial_state_distributed,
)

rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0")))
world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
coord = os.environ["JAX_COORDINATOR_ADDRESS"]
local_device_ids_env = os.environ.get("VALIDATE_LOCAL_DEVICE_IDS", "0").strip()
if local_device_ids_env.lower() in {"", "none", "all"}:
    local_device_ids = None
else:
    local_device_ids = [int(item.strip()) for item in local_device_ids_env.split(",") if item.strip()]

print(
    f"rank={rank}/{world} host={socket.gethostname()} coord={coord} "
    f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"local_device_ids={local_device_ids}",
    flush=True,
)

jax.distributed.initialize(
    coordinator_address=coord,
    num_processes=world,
    process_id=rank,
    local_device_ids=local_device_ids,
)
transpose_backend_name = os.environ.get("VALIDATE_TRANSPOSE_BACKEND", "MPI_A2A").upper()
fft_backend = os.environ.get("VALIDATE_FFT_BACKEND", "cudecomp")
input_dtype_name = os.environ.get("VALIDATE_INPUT_DTYPE", "complex64")
fft_norm = os.environ.get("VALIDATE_FFT_NORM", "backward")
transpose_backend = {
    "NCCL": jaxdecomp.TRANSPOSE_COMM_NCCL,
    "MPI_A2A": jaxdecomp.TRANSPOSE_COMM_MPI_A2A,
    "MPI_P2P": jaxdecomp.TRANSPOSE_COMM_MPI_P2P,
}[transpose_backend_name]
jaxdecomp.config.update("transpose_comm_backend", transpose_backend)
axis_contiguous = os.environ.get("VALIDATE_AXIS_CONTIGUOUS")
if axis_contiguous not in (None, ""):
    jaxdecomp.config.update(
        "transpose_axis_contiguous",
        axis_contiguous.strip().lower() not in {"0", "false", "no", "off"},
    )
if os.environ.get("VALIDATE_EXPLICIT_JAXDECOMP_INIT", "0") == "1":
    jaxdecomp.init()

pdims_env = os.environ.get("VALIDATE_PDIMS")
if pdims_env:
    pdims = tuple(int(item) for item in pdims_env.split(","))
else:
    pdims = (1, world)
mesh_devices = mesh_utils.create_device_mesh(pdims)
if os.environ.get("VALIDATE_TRANSPOSE_DEVICES", "0") == "1":
    mesh_devices = mesh_devices.T
axis_names = tuple(os.environ.get("VALIDATE_AXIS_NAMES", "x,y").split(","))
spec_mode = os.environ.get("VALIDATE_SPEC_MODE", "long").lower()
mesh = jax.make_mesh(
    mesh_devices.shape,
    axis_names=axis_names,
    devices=mesh_devices.flatten(),
    axis_types=(AxisType.Auto, AxisType.Auto),
)
if spec_mode == "short":
    sharding = NamedSharding(mesh, P(*axis_names))
elif spec_mode == "long":
    sharding = NamedSharding(mesh, P(*axis_names, None))
else:
    raise ValueError(f"unknown VALIDATE_SPEC_MODE={spec_mode!r}")

input_dtype = {
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
}[input_dtype_name]
host = np.ones((32, 32, 32), dtype=input_dtype)
callback_mode = os.environ.get("VALIDATE_CALLBACK_MODE", "numpy_slice")
workload = os.environ.get("VALIDATE_WORKLOAD", "fft").lower()
if callback_mode == "numpy_slice":
    data_callback = lambda index: np.ascontiguousarray(host[index])
elif callback_mode == "jax_ones":
    local_shape = (
        host.shape[0] // mesh.shape[axis_names[0]],
        host.shape[1] // mesh.shape[axis_names[1]],
        host.shape[2],
    )
    data_callback = lambda index: jnp.ones(local_shape, dtype=input_dtype)
else:
    raise ValueError(f"unknown VALIDATE_CALLBACK_MODE={callback_mode!r}")
arr = jax.make_array_from_callback(host.shape, sharding, data_callback)
expected = np.array(process_allgather(arr, tiled=True))

if workload == "fft":
    k = jaxdecomp.pfft3d(arr, norm=fft_norm, backend=fft_backend)
    k.block_until_ready()
    out = jaxdecomp.pifft3d(k, norm=fft_norm, backend=fft_backend)
    out.block_until_ready()
    reconstructed = np.array(process_allgather(out, tiled=True))
    local_arr = np.asarray(arr.addressable_shards[0].data)
    local_out = np.asarray(out.addressable_shards[0].data)
    err = float(np.max(np.abs(reconstructed - expected)))
    print(
        f"rank={rank} devices={jax.devices()} pdims={pdims} "
        f"local_device_count={jax.local_device_count()} "
        f"fft_backend={fft_backend} "
        f"fft_norm={fft_norm} "
        f"input_dtype={input_dtype_name} "
        f"transpose_backend={transpose_backend_name} "
        f"axis_contiguous={jaxdecomp.config.transpose_axis_contiguous} "
        f"axis_names={axis_names} spec_mode={spec_mode} "
        f"callback_mode={callback_mode} workload={workload} "
        f"arr_spec={arr.sharding.spec} k_spec={k.sharding.spec} out_spec={out.sharding.spec} "
        f"expected_shape={expected.shape} reconstructed_shape={reconstructed.shape} "
        f"expected_minmax=({float(np.min(expected.real))},{float(np.max(expected.real))}) "
        f"reconstructed_minmax=({float(np.min(reconstructed.real))},{float(np.max(reconstructed.real))}) "
        f"local_arr_shape={local_arr.shape} local_out_shape={local_out.shape} "
        f"local_arr_minmax=({float(np.min(local_arr.real))},{float(np.max(local_arr.real))}) "
        f"local_out_minmax=({float(np.min(local_out.real))},{float(np.max(local_out.real))}) "
        f"roundtrip_err={err}",
        flush=True,
    )
elif workload == "lpt":
    field_sharding = NamedSharding(mesh, P(*axis_names, None))
    disp_sharding = NamedSharding(mesh, P(*axis_names, None, None))
    res = host.shape[0]
    boxsize = 100.0

    grid = np.indices(host.shape, dtype=np.float32)
    delta_host = (
        0.01 * np.sin(2.0 * np.pi * grid[0] / res)
        + 0.02 * np.cos(4.0 * np.pi * grid[1] / res)
        + 0.015 * np.sin(6.0 * np.pi * grid[2] / res)
    ).astype(np.float32)
    delta_host -= delta_host.mean(dtype=np.float32)
    delta = jax.make_array_from_callback(
        delta_host.shape,
        field_sharding,
        lambda index: np.ascontiguousarray(delta_host[index]),
    )

    @jax.jit
    def _fphi_from_delta(delta_arr):
        fdelta = jaxdecomp.pfft3d(delta_arr.astype(jnp.complex64), norm="backward", backend=fft_backend)
        fdelta = lax.with_sharding_constraint(fdelta, field_sharding)
        k_vecs = build_k_vecs_dist(fdelta, boxsize=boxsize, res=res)
        ksquare = sum(ki**2 for ki in k_vecs)
        mask = (ksquare != 0).astype(jnp.float32)
        ksquare = ksquare.at[0, 0, 0].set(jnp.float32(1.0))
        ksquare = lax.with_sharding_constraint(ksquare, field_sharding)
        inv_lap = (-1.0 / ksquare) * mask
        return lax.with_sharding_constraint(inv_lap * fdelta, field_sharding)

    fphi = _fphi_from_delta(delta)
    fphi.block_until_ready()
    psi, mom = compute_2lpt_initial_state_distributed(
        fphi,
        res=res,
        boxsize=boxsize,
        Dplus=1.0 / 32.0,
        grad_kernel_order=4,
        dtype_num=32,
        dtype_c_num=64,
        no_factors=False,
        field_sharding=field_sharding,
        disp_sharding=disp_sharding,
        fft_sharding=getattr(fphi, "sharding", None),
        fft_backend=fft_backend,
        progress=False,
    )
    psi.block_until_ready()
    mom.block_until_ready()
    psi_local = np.asarray(psi.addressable_shards[0].data)
    mom_local = np.asarray(mom.addressable_shards[0].data)
    finite = bool(np.isfinite(psi_local).all() and np.isfinite(mom_local).all())
    err = 0.0 if finite else 1.0
    print(
        f"rank={rank} devices={jax.devices()} pdims={pdims} "
        f"local_device_count={jax.local_device_count()} "
        f"fft_backend={fft_backend} transpose_backend={transpose_backend_name} "
        f"axis_contiguous={jaxdecomp.config.transpose_axis_contiguous} "
        f"axis_names={axis_names} spec_mode={spec_mode} workload={workload} "
        f"delta_spec={delta.sharding.spec} fphi_spec={fphi.sharding.spec} "
        f"psi_spec={psi.sharding.spec} mom_spec={mom.sharding.spec} "
        f"psi_local_shape={psi_local.shape} mom_local_shape={mom_local.shape} "
        f"psi_absmax={float(np.max(np.abs(psi_local)))} "
        f"mom_absmax={float(np.max(np.abs(mom_local)))} "
        f"finite={finite}",
        flush=True,
    )
else:
    raise ValueError(f"unknown VALIDATE_WORKLOAD={workload!r}")
sync_global_devices("cudecomp-smoke")
if err > 1e-5:
    raise RuntimeError(f"{fft_backend} {workload} validation error too large: {err}")
if rank == 0:
    print(f"multinode_{fft_backend}_{workload}_ok", flush=True)
