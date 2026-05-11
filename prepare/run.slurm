#!/bin/bash
#SBATCH --job-name=gen_dmo
#SBATCH --partition=ghx4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --account=bdne-dtai-gh    # <- match to a "Project" returned by the "accounts" command #SBATCH --constraint=work
#SBATCH --no-requeue
#SBATCH -t 00:10:00
#SBATCH -e %x.o%j
#SBATCH -o %x.o%j
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest   # select a cpu close to gpu on pci bus topology  

module purge             
#module load gcc-native/12.3
module load cuda/12.4.0
module load cudnn/9.3.0.75
module load gsl
module load nccl/2.19.1.awsplugin

conda activate pm
which python

#nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
#           --format=csv --loop-ms=500 > gpu_usage_${SLURM_JOB_ID}.log &
#GPU_MONITOR_PID=$!

unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/sw/user/gsl/lib



python pm.py 0 1

#kill $GPU_MONITOR_PID

