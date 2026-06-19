#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -C h100
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --job-name=genhalos3gpc_sim0_16gpu
#SBATCH --output=/mnt/ceph/users/spandey/quijote_v2_gotham/IC_3gpc_test/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/quijote_v2_gotham/IC_3gpc_test/logs/%x.%j.err

set -euo pipefail

ROOT=/mnt/ceph/users/spandey/quijote_v2_gotham
SCRIPT=${ROOT}/IC_3gpc_test/sample_halos_3gpc_multigpu.py
SIM_ID="${SIM_ID:-0}"

mkdir -p "${ROOT}/IC_3gpc_test/logs"
mkdir -p "${ROOT}/IC_3gpc_test/generated_halo_cats"

echo "--- JOB DETAILS ---"
echo "Job Name: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Nodes: ${SLURM_NNODES}"
echo "Tasks: ${SLURM_NTASKS} (one sampler process per GPU)"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-4}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "SIM_ID: ${SIM_ID}"
echo "Node list: ${SLURM_NODELIST}"
echo "-------------------"

module load nvhpc/25.1
module load python
source ~/miniconda3/bin/activate discodj

# Keep the launch style consistent with the distributed PM scripts.  The
# sampler itself does not use MPI collectives; mpirun is only used here to
# create one independent Python process per GPU with stable local rank binding.
NVHPC_STACK="${NVHPC_ROOT}/Linux_x86_64/25.1"
OMPI_HOME="${NVHPC_STACK}/comm_libs/12.6/hpcx/hpcx-2.21/ompi"
export OMPI_HOME
export OPAL_PREFIX="${OMPI_HOME}"
export OPAL_DATA_PATH="${OMPI_HOME}/share"
export CUDA_HOME="${NVHPC_STACK}/cuda/12.6"
export CUDA_ROOT="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${OMPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NVHPC_STACK}/comm_libs/12.6/nccl/lib:${NVHPC_STACK}/math_libs/12.6/lib64:${NVHPC_STACK}/cuda/12.6/lib64:${OMPI_HOME}/lib:${NVHPC_STACK}/compilers/lib:${LD_LIBRARY_PATH:-}"

unset NCCL_NET_PLUGIN
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_CPP_MIN_LOG_LEVEL=2

# Sampling controls.  Increase DISCO_GEN_BATCH_SIZE if nvitop shows large free
# memory and throughput improves; lower it if torch reports CUDA OOM.
export DISCO_GEN_RUN_ID="${DISCO_GEN_RUN_ID:-${SLURM_JOB_ID}}"
export DISCO_GEN_OUTPUT_DIR="${DISCO_GEN_OUTPUT_DIR:-${ROOT}/IC_3gpc_test/generated_halo_cats}"
export DISCO_GEN_BATCH_SIZE="${DISCO_GEN_BATCH_SIZE:-8192}"
export DISCO_GEN_COMPILE="${DISCO_GEN_COMPILE:-1}"
export DISCO_GEN_PROGRESS="${DISCO_GEN_PROGRESS:-1}"
export DISCO_GEN_LOG_EVERY="${DISCO_GEN_LOG_EVERY:-5}"
export DISCO_GEN_SEED="${DISCO_GEN_SEED:-12345}"
export DISCO_GEN_OVERWRITE="${DISCO_GEN_OVERWRITE:-0}"
export DISCO_GEN_KEEP_PARTIALS="${DISCO_GEN_KEEP_PARTIALS:-0}"
export DISCO_GEN_USE_PARAM_FILE="${DISCO_GEN_USE_PARAM_FILE:-0}"
export DISCO_GEN_CHECKPOINT="${DISCO_GEN_CHECKPOINT:-${ROOT}/GOTHAM/checkpoints/checkpoint_new.pt}"
export DISCO_GEN_PARAM_FILE="${DISCO_GEN_PARAM_FILE:-${ROOT}/GOTHAM/checkpoints/quijote_params.txt}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "SCRIPT=${SCRIPT}"
echo "DISCO_GEN_OUTPUT_DIR=${DISCO_GEN_OUTPUT_DIR}"
echo "DISCO_GEN_BATCH_SIZE=${DISCO_GEN_BATCH_SIZE}"
echo "DISCO_GEN_COMPILE=${DISCO_GEN_COMPILE}"
echo "DISCO_GEN_LOG_EVERY=${DISCO_GEN_LOG_EVERY}"
echo "DISCO_GEN_OVERWRITE=${DISCO_GEN_OVERWRITE}"
echo "DISCO_GEN_USE_PARAM_FILE=${DISCO_GEN_USE_PARAM_FILE}"
echo "DISCO_GEN_CHECKPOINT=${DISCO_GEN_CHECKPOINT}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

cd "${ROOT}"
export SCRIPT SIM_ID
time mpirun -np "${SLURM_NTASKS}" --map-by "ppr:4:node:PE=${SLURM_CPUS_PER_TASK}" --bind-to core \
    bash -c 'export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK}"; export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor_${SLURM_JOB_ID}_${OMPI_COMM_WORLD_RANK}"; mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"; python -u "${SCRIPT}" "${SIM_ID}"'

echo "Halo sampling for SIM_ID=${SIM_ID} complete"
