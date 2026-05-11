#!/bin/bash

# --- Configuration ---
# Set the parameters for your runs
TOTAL_DEVICES=10

# Set the base directory
WORK_DIR="/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM"

# Set the directory to store the generated Slurm scripts and logs
SCRIPT_DIR="${WORK_DIR}/run_scripts_gen"
LOG_DIR="${SCRIPT_DIR}/logs"

# File containing simulation IDs
# SIM_ID_FILE="${SCRIPT_DIR}/rand_sel_cosmo.txt"
SIM_ID_FILE="${SCRIPT_DIR}/all_LH.txt"

# --- Script Logic ---

# Create the directories if they don't exist
mkdir -p "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}"

# Read simulation IDs from file into an array
mapfile -t SIM_IDS < "${SIM_ID_FILE}"
TOTAL_SIMS=${#SIM_IDS[@]}

# Calculate simulations per device
SIMS_PER_DEVICE=$(( (TOTAL_SIMS + TOTAL_DEVICES - 1) / TOTAL_DEVICES ))

echo "Configuration:"
echo "TOTAL_DEVICES:  ${TOTAL_DEVICES}"
echo "Total Sims:     ${TOTAL_SIMS}"
echo "Sims per Device: ${SIMS_PER_DEVICE}"
echo "Working Dir:    ${WORK_DIR}"
echo "Script Dir:     ${SCRIPT_DIR}"
echo "Log Dir:        ${LOG_DIR}"
echo "--------------------------------"

# Loop from 0 to TOTAL_DEVICES - 1
for (( JDEVICE=0; JDEVICE<TOTAL_DEVICES; JDEVICE++ )); do
    # Calculate array indices for this device
    ARRAY_START=$((SIMS_PER_DEVICE * JDEVICE))
    ARRAY_END=$((SIMS_PER_DEVICE * (JDEVICE + 1)))

    # Don't exceed total number of simulations
    if (( ARRAY_END > TOTAL_SIMS )); then
        ARRAY_END=${TOTAL_SIMS}
    fi

    # Extract simulation IDs for this device
    DEVICE_SIMS=("${SIM_IDS[@]:ARRAY_START:$((ARRAY_END - ARRAY_START))}")

    # Skip if no simulations for this device
    if (( ${#DEVICE_SIMS[@]} == 0 )); then
        echo "Device ${JDEVICE}: No simulations assigned, skipping"
        continue
    fi

    # Convert array to space-separated string for passing to script
    DEVICE_SIMS_STR="${DEVICE_SIMS[*]}"

    # Define unique names for the job and the script file
    JOB_NAME="pm_${JDEVICE}_nsims_${#DEVICE_SIMS[@]}"
    SLURM_SCRIPT_PATH="${SCRIPT_DIR}/${JOB_NAME}.slurm"

    echo "Generating script: ${SLURM_SCRIPT_PATH} (Device ${JDEVICE}: ${#DEVICE_SIMS[@]} sims)"

    # Use a "Here Document" to write the Slurm script
    cat > "${SLURM_SCRIPT_PATH}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -C h100
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=10:30:00
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x.%j.out
#SBATCH --error=${LOG_DIR}/%x.%j.err

echo "--- JOB DETAILS ---"
echo "Job Name: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Running on: \$(hostname)"
echo "JDEVICE: ${JDEVICE}"
echo "Simulation IDs: ${DEVICE_SIMS_STR}"
echo "Number of sims: ${#DEVICE_SIMS[@]}"
echo "-------------------"

source /etc/profile.d/modules.sh
module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi

# Array of simulation IDs for this device
SIM_ARRAY=(${DEVICE_SIMS_STR})

# Loop through simulation IDs for this device
for i in "\${SIM_ARRAY[@]}";
do
    echo \$i;
    cd "/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/run_scripts_gen";
    echo "\$PWD";
    time srun python /mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/src/generate_catalog.py \$i;
    echo "done";
done

echo "All runs complete for device ${JDEVICE}"
EOF

    # Submit the generated script to Slurm
    sbatch "${SLURM_SCRIPT_PATH}"

done

echo "--------------------------------"
echo "All ${TOTAL_DEVICES} jobs have been submitted."
