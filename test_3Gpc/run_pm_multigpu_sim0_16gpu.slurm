#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -C h100
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=pm3gpc_sim0_16gpu
#SBATCH --output=/mnt/ceph/users/spandey/quijote_v2_gotham/IC_3gpc_test/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/quijote_v2_gotham/IC_3gpc_test/logs/%x.%j.err

set -euo pipefail

ROOT=/mnt/ceph/users/spandey/quijote_v2_gotham
SCRIPT=${ROOT}/IC_3gpc_test/pm_multigpu.py
SIM_ID=0

mkdir -p "${ROOT}/IC_3gpc_test/logs"
mkdir -p "${ROOT}/IC_3gpc_test/dmo/disco"

echo "--- JOB DETAILS ---"
echo "Job Name: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS} (one JAX process per GPU)"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-4}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "SIM_ID: ${SIM_ID}"
echo "Node list: ${SLURM_NODELIST}"
echo "-------------------"

module load nvhpc/25.1
module load python
source ~/miniconda3/bin/activate discodj

# Runtime libraries required by the CUDA/cuDecomp-enabled jaxDecomp extension.
NVHPC_STACK="${NVHPC_ROOT}/Linux_x86_64/25.1"
OMPI_HOME="${NVHPC_STACK}/comm_libs/12.6/hpcx/hpcx-2.21/ompi"
export OMPI_HOME
export OPAL_PREFIX="${OMPI_HOME}"
export OPAL_DATA_PATH="${OMPI_HOME}/share"
export CUDA_HOME="${NVHPC_STACK}/cuda/12.6"
export CUDA_ROOT="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${OMPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NVHPC_STACK}/comm_libs/12.6/nccl/lib:${NVHPC_STACK}/math_libs/12.6/lib64:${NVHPC_STACK}/cuda/12.6/lib64:${OMPI_HOME}/lib:${NVHPC_STACK}/compilers/lib:${LD_LIBRARY_PATH:-}"

# JAX/cuDecomp distributed: one process per GPU.  cuDecomp requires
# product(pdims) == MPI rank count, so do not run cudecomp with one process
# owning multiple local GPUs.
export JAX_NUM_PROCESSES="${SLURM_NTASKS}"
export JAX_LOCAL_DEVICE_IDS=0
export JAX_COORDINATOR_HOST="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
export JAX_COORDINATOR_PORT="${JAX_COORDINATOR_PORT:-12355}"
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_HOST}:${JAX_COORDINATOR_PORT}"

# Required production setting: factor=2 -> res_pm=3072.
export DISCO_PM_FACTOR=2
export DISCO_HALO_SIZE=72
# Keep IC/2LPT on the square pencil expected by jaxDecomp full FFTs, then
# reshard only the PM state to the X-slab required by the memory-saving RFFT.
export DISCO_LPT_PDIMS="${DISCO_LPT_PDIMS:-4,4}"
export DISCO_PM_PDIMS="${DISCO_PM_PDIMS:-16,1}"

# Runtime environment.
unset NCCL_NET_PLUGIN
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Keep the 16-GPU run consistent with the 8-GPU launcher.  The platform
# allocator releases freed temporaries instead of keeping them in JAX's BFC
# cache, so nvitop is closer to the live memory footprint and large FFT
# temporaries are less likely to fragment the GPU heap.
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# Verbose progress logging for long distributed LPT/PM stages.
export DISCO_PROGRESS_2LPT="${DISCO_PROGRESS_2LPT:-1}"
export DISCO_PROGRESS_SYNC="${DISCO_PROGRESS_SYNC:-1}"
export DISCO_DISTRIBUTED_CPU_POSTPROCESS="${DISCO_DISTRIBUTED_CPU_POSTPROCESS:-1}"
export DISCO_CPU_REDUCE_BACKEND="${DISCO_CPU_REDUCE_BACKEND:-socket}"
export DISCO_CPU_REDUCE_PORT="${DISCO_CPU_REDUCE_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export DISCO_CPU_REDUCE_PORT_SPAN="${DISCO_CPU_REDUCE_PORT_SPAN:-2000}"
export DISCO_CPU_REDUCE_CHUNK_MB="${DISCO_CPU_REDUCE_CHUNK_MB:-256}"

# FFT backend controls.  LPT defaults to cuDecomp because this environment now
# has a CUDA/cuDecomp-enabled jaxDecomp build.
export DISCO_IC_FFT_BACKEND="${DISCO_IC_FFT_BACKEND:-JAX}"
export DISCO_LPT_FFT_BACKEND="${DISCO_LPT_FFT_BACKEND:-cudecomp}"
export DISCO_LPT_MU2_FFT_BACKEND="${DISCO_LPT_MU2_FFT_BACKEND:-JAX}"
export DISCO_PM_FFT_BACKEND="${DISCO_PM_FFT_BACKEND:-JAX_RFFT}"
export DISCO_JAXDECOMP_TRANSPOSE_BACKEND="${DISCO_JAXDECOMP_TRANSPOSE_BACKEND:-NCCL}"
export DISCO_FFT_PREFLIGHT="${DISCO_FFT_PREFLIGHT:-1}"

echo "JAX_COORDINATOR_ADDRESS=${JAX_COORDINATOR_ADDRESS}"
echo "JAX_LOCAL_DEVICE_IDS=${JAX_LOCAL_DEVICE_IDS}"
echo "DISCO_PM_FACTOR=${DISCO_PM_FACTOR}"
echo "DISCO_HALO_SIZE=${DISCO_HALO_SIZE}"
echo "DISCO_LPT_PDIMS=${DISCO_LPT_PDIMS}"
echo "DISCO_PM_PDIMS=${DISCO_PM_PDIMS}"
echo "XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR}"
echo "DISCO_PROGRESS_2LPT=${DISCO_PROGRESS_2LPT}"
echo "DISCO_PROGRESS_SYNC=${DISCO_PROGRESS_SYNC}"
echo "DISCO_DISTRIBUTED_CPU_POSTPROCESS=${DISCO_DISTRIBUTED_CPU_POSTPROCESS}"
echo "DISCO_CPU_REDUCE_BACKEND=${DISCO_CPU_REDUCE_BACKEND}"
echo "DISCO_CPU_REDUCE_PORT=${DISCO_CPU_REDUCE_PORT}"
echo "DISCO_CPU_REDUCE_PORT_SPAN=${DISCO_CPU_REDUCE_PORT_SPAN}"
echo "DISCO_CPU_REDUCE_CHUNK_MB=${DISCO_CPU_REDUCE_CHUNK_MB}"
echo "DISCO_IC_FFT_BACKEND=${DISCO_IC_FFT_BACKEND}"
echo "DISCO_LPT_FFT_BACKEND=${DISCO_LPT_FFT_BACKEND}"
echo "DISCO_LPT_MU2_FFT_BACKEND=${DISCO_LPT_MU2_FFT_BACKEND}"
echo "DISCO_PM_FFT_BACKEND=${DISCO_PM_FFT_BACKEND}"
echo "DISCO_JAXDECOMP_TRANSPOSE_BACKEND=${DISCO_JAXDECOMP_TRANSPOSE_BACKEND}"
echo "DISCO_FFT_PREFLIGHT=${DISCO_FFT_PREFLIGHT}"
echo "OMPI_HOME=${OMPI_HOME}"
echo "OPAL_PREFIX=${OPAL_PREFIX}"
echo "CUDA_HOME=${CUDA_HOME}"

cd "${ROOT}"
export SCRIPT SIM_ID
time mpirun -np "${SLURM_NTASKS}" --map-by "ppr:4:node:PE=${SLURM_CPUS_PER_TASK}" --bind-to core \
    bash -c 'export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK}"; export JAX_LOCAL_DEVICE_IDS=0; python -u "${SCRIPT}" "${SIM_ID}"'

echo "Simulation ${SIM_ID} complete"
