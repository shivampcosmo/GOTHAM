#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=8:00:00
#SBATCH --job-name=ng16_wposembed
#SBATCH --partition=ghx4
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=/projects/bdne/spandey3/FINAL_GOTHAM/GOTHAM/run_scripts/logs/%x.%j.out
#SBATCH --error=/projects/bdne/spandey3/FINAL_GOTHAM/GOTHAM/run_scripts/logs/%x.%j.err


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
conda activate /u/spandey3/gotham2
# which python
module load cuda/12.4.0
module load cudnn/9.3.0.75
module load gsl

# module load nccl/2.19.1.awsplugin
# module load cudatoolkit/24.3_12.3


master_node=$SLURMD_NODENAME

cd /projects/bdne/spandey3/FINAL_GOTHAM/GOTHAM/src/
srun --export=ALL python `which torchrun` \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $SLURM_GPUS_PER_NODE \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $master_node:29500 \
        train_dtai_v2.py --grid_sbox=16 --add_space_token=False --subsel_type=all --learning_rate=0.001 --max_iters=500 --n_embd=192 --patch_size=2 --loss_type=cross_entropy --cnn_type=vit
echo "done"
