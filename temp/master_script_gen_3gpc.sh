#!/bin/bash

# --- Configuration ---
SIMID=${1:-663}
NUM_WORKERS=8

WORK_DIR="/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/temp"
LOG_DIR="${WORK_DIR}/logs_3gpc"
mkdir -p "${LOG_DIR}"

echo "=== 3Gpc Halo Catalog Generation ==="
echo "SIMID:       ${SIMID}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "======================================"

# Array to collect job IDs for dependency
JOB_IDS=()

# Submit one SLURM job per GPU worker
for (( W=0; W<NUM_WORKERS; W++ )); do
    JOB_NAME="gen3gpc_sim${SIMID}_w${W}"
    SLURM_SCRIPT="${WORK_DIR}/${JOB_NAME}.slurm"

    cat > "${SLURM_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C h100
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x.%j.out
#SBATCH --error=${LOG_DIR}/%x.%j.err

echo "--- JOB DETAILS ---"
echo "Job Name: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Running on: \$(hostname)"
echo "Worker: ${W}/${NUM_WORKERS}"
echo "SimID: ${SIMID}"
echo "-------------------"

source /etc/profile.d/modules.sh
module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi

time srun python ${WORK_DIR}/generate_3gpc_halos_worker.py \\
    --simid ${SIMID} \\
    --worker_id ${W} \\
    --num_workers ${NUM_WORKERS}

echo "Worker ${W} complete"
EOF

    # Submit and capture job ID
    JID=$(sbatch --parsable "${SLURM_SCRIPT}")
    echo "Submitted worker ${W}: job ${JID}"
    JOB_IDS+=("${JID}")
done

# Build dependency string (afterok:id1:id2:...)
DEP_STR="afterok"
for JID in "${JOB_IDS[@]}"; do
    DEP_STR="${DEP_STR}:${JID}"
done

# Submit combine job that runs after all workers finish
COMBINE_JOB_NAME="gen3gpc_combine_sim${SIMID}"
COMBINE_SCRIPT="${WORK_DIR}/${COMBINE_JOB_NAME}.slurm"

cat > "${COMBINE_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=784G
#SBATCH -C rome
#SBATCH -p cmbas
#SBATCH --time=01:00:00
#SBATCH --job-name=${COMBINE_JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x.%j.out
#SBATCH --error=${LOG_DIR}/%x.%j.err

echo "--- COMBINE JOB ---"
echo "Job Name: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Running on: \$(hostname)"
echo "SimID: ${SIMID}"
echo "-------------------"

source /etc/profile.d/modules.sh
module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

time srun python ${WORK_DIR}/combine_3gpc_halos.py \\
    --simid ${SIMID} \\
    --num_workers ${NUM_WORKERS}

echo "Combine complete"
EOF

COMBINE_JID=$(sbatch --parsable --dependency="${DEP_STR}" "${COMBINE_SCRIPT}")
echo ""
echo "Submitted combine job: ${COMBINE_JID} (depends on: ${JOB_IDS[*]})"
echo "======================================"
echo "All jobs submitted. Monitor with: squeue -u \$USER"
