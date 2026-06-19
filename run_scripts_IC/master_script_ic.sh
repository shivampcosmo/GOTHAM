#!/bin/bash
set -euo pipefail

# Generate and submit 2LPT IC creation jobs for the new LH cosmologies.
# This covers sim ids [2000, 4000), split into 20 jobs of 100 sims each.

# --- Configuration ---
SIM_START=2000
SIM_END=4000
SIMS_PER_JOB=100
DEVICE_START=20

WORK_DIR="/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM"
SCRIPT_DIR="${WORK_DIR}/run_scripts_IC"
LOG_DIR="${WORK_DIR}/run_scripts/logs"

LH_ROOT="/mnt/ceph/users/spandey/discodj_runs/LH"
TWOLPT_BIN="/mnt/ceph/users/spandey/fastpm/paco_2lpt/2LPTic"
TWOLPT_PARAM="2LPT_512.param"

# Set SUBMIT_JOBS=0 when you only want to generate scripts without sbatch.
SUBMIT_JOBS="${SUBMIT_JOBS:-1}"

# --- Script Logic ---
mkdir -p "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}"

if (( SIM_END <= SIM_START )); then
    echo "SIM_END must be greater than SIM_START" >&2
    exit 1
fi

TOTAL_SIMS=$((SIM_END - SIM_START))
TOTAL_JOBS=$(((TOTAL_SIMS + SIMS_PER_JOB - 1) / SIMS_PER_JOB))

echo "Configuration:"
echo "SIM range:      ${SIM_START}..$((SIM_END - 1))"
echo "SIMS_PER_JOB:   ${SIMS_PER_JOB}"
echo "TOTAL_JOBS:     ${TOTAL_JOBS}"
echo "DEVICE_START:   ${DEVICE_START}"
echo "Working Dir:    ${WORK_DIR}"
echo "Script Dir:     ${SCRIPT_DIR}"
echo "Log Dir:        ${LOG_DIR}"
echo "Submit Jobs:    ${SUBMIT_JOBS}"
echo "--------------------------------"

for (( JOB_INDEX=0; JOB_INDEX<TOTAL_JOBS; JOB_INDEX++ )); do
    JDEVICE=$((DEVICE_START + JOB_INDEX))
    I_START=$((SIM_START + SIMS_PER_JOB * JOB_INDEX))
    I_END=$((I_START + SIMS_PER_JOB))
    if (( I_END > SIM_END )); then
        I_END=${SIM_END}
    fi

    JOB_NAME="genic_${JDEVICE}_${I_START}_${I_END}"
    SLURM_SCRIPT_PATH="${SCRIPT_DIR}/${JOB_NAME}.slurm"

    echo "Generating script: ${SLURM_SCRIPT_PATH} (i: ${I_START}..$((I_END - 1)))"

    cat > "${SLURM_SCRIPT_PATH}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -C rome
#SBATCH -p cmbas
#SBATCH --time=1:00:00
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x.%j.out
#SBATCH --error=${LOG_DIR}/%x.%j.err

set -euo pipefail

echo "--- JOB DETAILS ---"
echo "Job Name: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Running on: \$(hostname)"
echo "JDEVICE: ${JDEVICE}"
echo "I_START: ${I_START}"
echo "I_END: ${I_END}"
echo "-------------------"

module --force purge
module load modules/2.4-20250724 openmpi/4.1.8 gsl/2.7.1 fftw/mpi-2.1.5

for (( i=${I_START}; i<${I_END}; i++ )); do
    echo "\${i}"
    cd "${LH_ROOT}/\${i}/ICs"
    echo "\${PWD}"

    if [[ ! -f "${TWOLPT_PARAM}" ]]; then
        echo "Missing ${TWOLPT_PARAM} in \${PWD}" >&2
        exit 1
    fi

    time srun "${TWOLPT_BIN}" "${TWOLPT_PARAM}"
    echo "done"
done

echo "All runs complete for device ${JDEVICE}"
EOF

    chmod +x "${SLURM_SCRIPT_PATH}"

    if [[ "${SUBMIT_JOBS}" == "1" ]]; then
        sbatch "${SLURM_SCRIPT_PATH}"
    fi
done

echo "--------------------------------"
if [[ "${SUBMIT_JOBS}" == "1" ]]; then
    echo "All ${TOTAL_JOBS} jobs have been submitted."
else
    echo "Generated ${TOTAL_JOBS} job scripts without submitting."
fi
