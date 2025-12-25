#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=50
#SBATCH -C rome
#SBATCH -p cmbas
#SBATCH --time=1:00:00
#SBATCH --job-name=convert_ICs
#SBATCH --output=/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/ICs/logs/convert_ICs_%j.out

# Start from an "empty" module collection.
# module purge
# module load python
# source ~/miniconda3/bin/activate ili-sbi

module purge
module load openmpi/4.1.8
module load python
source ~/miniconda3/bin/activate discodj

# Change to the ICs directory
cd /mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/ICs

# Run the convert_ICs script with 50 MPI processes
echo "Starting convert_ICs with 50 parallel processes"
echo "Working directory: $PWD"
srun --cpu-bind=cores python convert_ICs.py

echo "Conversion complete"
