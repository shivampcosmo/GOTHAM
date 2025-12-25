#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH -C rome
#SBATCH -p cmbas
#SBATCH --time=1:00:00
#SBATCH --job-name=genic
#SBATCH --output=/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/run_scripts/logs/genic_%j.out

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630  gcc/11.2.0  openmpi/1.10.7 slurm gcc openmpi gsl fftw/mpi-2.1.5 hdf5/mpi-1.10.8


# We assume this executable is in the directory from which you ran sbatch.
for i in {0..10};
do
    echo $i;
    cd "/mnt/ceph/users/spandey/discodj_runs/LH/${i}/ICs";
    echo "$PWD";
    time srun /mnt/ceph/users/spandey/fastpm/paco_2lpt/2LPTic 2LPT.param;
    echo "done";
done
