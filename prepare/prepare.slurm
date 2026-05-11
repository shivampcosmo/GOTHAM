#!/bin/bash
#SBATCH --job-name=select_20000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --partition=cpu 
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j

# --- conda (Miniforge, HPC-safe) ---
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
#conda activate hod
conda activate general

which python
echo "Start time: $(date)"

# --- MPI via Slurm ---
#srun python prepare/process_DMO_fields.py
#srun python prepare/select_20000.py
#mpirun -np 4 python prepare/process_halo_props_quijote.py
#python prepare/process_halo_props.py

#srun python prepare/get_LH_HOD_wide_flat.py
#python prepare/prepare_fastpm.py 0
#python prepare/prepare_fastpm_corr.py 0
python prepare/select_20000.py


echo "End time: $(date)"