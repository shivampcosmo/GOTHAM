#!/bin/bash

# --- Configuration ---
# Set the parameters for your runs
TOTAL_DEVICES=20

# Set the base directory
WORK_DIR="/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM"

# Set the directory to store the generated Slurm scripts and logs
SCRIPT_DIR="${WORK_DIR}/run_scripts"
LOG_DIR="${SCRIPT_DIR}/logs"

# --- Script Logic ---

# Create the directories if they don't exist
mkdir -p "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Configuration:"
echo "TOTAL_DEVICES:  ${TOTAL_DEVICES}"
echo "Working Dir:    ${WORK_DIR}"
echo "Script Dir:     ${SCRIPT_DIR}"
echo "Log Dir:        ${LOG_DIR}"
echo "--------------------------------"

# Loop from 0 to TOTAL_DEVICES - 1
for (( JDEVICE=0; JDEVICE<TOTAL_DEVICES; JDEVICE++ )); do
    # Calculate i range for this device
    I_START=$((100 * JDEVICE))
    I_END=$((100 * (JDEVICE + 1)))

    # Define unique names for the job and the script file
    JOB_NAME="genic_${JDEVICE}_${I_START}_${I_END}"
    SLURM_SCRIPT_PATH="${SCRIPT_DIR}/${JOB_NAME}.slurm"

    echo "Generating script: ${SLURM_SCRIPT_PATH} (i: ${I_START}..${I_END})"

    # Use a "Here Document" to write the Slurm script
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

echo "--- JOB DETAILS ---"
echo "Job Name: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Running on: \$(hostname)"
echo "JDEVICE: ${JDEVICE}"
echo "I_START: ${I_START}"
echo "I_END: ${I_END}"
echo "-------------------"

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630  gcc/11.2.0  openmpi/1.10.7 slurm gcc openmpi gsl fftw/mpi-2.1.5 hdf5/mpi-1.10.8

# Loop through i values for this device
for i in {${I_START}..$((I_END - 1))};
do
    echo \$i;
    cd "/mnt/ceph/users/spandey/discodj_runs/LH/\${i}/ICs";
    echo "\$PWD";
    time srun /mnt/ceph/users/spandey/fastpm/paco_2lpt/2LPTic 2LPT.param;
    echo "done";
done

echo "All runs complete for device ${JDEVICE}"
EOF

    # Submit the generated script to Slurm
    sbatch "${SLURM_SCRIPT_PATH}"

done

echo "--------------------------------"
echo "All ${TOTAL_DEVICES} jobs have been submitted."
