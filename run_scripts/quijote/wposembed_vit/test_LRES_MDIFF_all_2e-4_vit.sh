#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --job-name=wposembed
#SBATCH --partition=ghx4
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=/projects/bdne/spandey3/halo_gotham/GOTHAM/run_scripts/quijote/logs/%x.%j.out
#SBATCH --error=/projects/bdne/spandey3/halo_gotham/GOTHAM/run_scripts/quijote/logs/%x.%j.err


# Setup variables for torchrun rdzv_endpoint
# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
# echo "Head node: $head_node"
# echo "Head node IP: $head_node_ip"

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
module load nccl

master_node=$SLURMD_NODENAME

cd /projects/bdne/spandey3/halo_gotham/GOTHAM/src/
srun --export=ALL python `which torchrun` \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $SLURM_GPUS_PER_NODE \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $master_node:29500 \
        train_dtai_LRES_MDIFF.py --grid_sbox=8 --add_space_token=False --subsel_type=all --learning_rate=0.0002 --max_iters=400 --loss_type=cross_entropy --cnn_type=vit
echo "done"
