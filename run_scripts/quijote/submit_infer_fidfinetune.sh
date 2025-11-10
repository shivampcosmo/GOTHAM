#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=15
#SBATCH --time=2:00:00
#SBATCH --job-name=infer_fidfinetune
#SBATCH --partition=ghx4
#SBATCH --mem=512G
#SBATCH --gpus-per-node=1
#SBATCH --output=/projects/bdne/spandey3/halo_gotham/GOTHAM/run_scripts/quijote/logs/%x.%j.out
#SBATCH --error=/projects/bdne/spandey3/halo_gotham/GOTHAM/run_scripts/quijote/logs/%x.%j.err


# export MASTER_PORT=12367
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load python
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/user/python/miniforge3-pytorch-2.5.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh" ]; then
        . "/sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh"
    else
        export PATH="/sw/user/python/miniforge3-pytorch-2.5.0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate /u/spandey3/gotham
which python
module load cuda

master_node=$SLURMD_NODENAME

cd /projects/bdne/spandey3/halo_gotham/GOTHAM/src/
srun --export=ALL python infer_dtai_fid.py
echo "done"