#!/bin/bash
set -euo pipefail

module load nvhpc/25.1
module load python
source ~/miniconda3/bin/activate discodj

ROOT=/mnt/ceph/users/spandey/quijote_v2_gotham
NVHPC_STACK="${NVHPC_ROOT}/Linux_x86_64/25.1"
OMPI_HOME="${NVHPC_STACK}/comm_libs/12.6/hpcx/hpcx-2.21/ompi"
export OMPI_HOME
export OPAL_PREFIX="${OMPI_HOME}"
export OPAL_DATA_PATH="${OMPI_HOME}/share"
export CUDA_HOME="${NVHPC_STACK}/cuda/12.6"
export CUDA_ROOT="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${OMPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NVHPC_STACK}/comm_libs/12.6/nccl/lib:${NVHPC_STACK}/math_libs/12.6/lib64:${NVHPC_STACK}/cuda/12.6/lib64:${OMPI_HOME}/lib:${NVHPC_STACK}/compilers/lib:${LD_LIBRARY_PATH:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export JAX_COORDINATOR_ADDRESS="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1):12407"

if [[ "$#" -gt 0 ]]; then
    exec "$@"
fi

python -u "${ROOT}/IC_3gpc_test/_validate_cudecomp_multinode.py"
