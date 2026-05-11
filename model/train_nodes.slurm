#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --job-name=tune_quijote_1872236
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=ghx4
#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=/u/yzhang116/logs/%x.o%j
#SBATCH --error=/u/yzhang116/logs/%x.o%j

module purge             
module load cuda/12.4.0
module load cudnn/9.3.0.75
module load nccl/2.19.1.awsplugin
module load cudatoolkit/24.3_12.3

# >>> conda initialize >>>
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate nn

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
    hostname -I | awk '{print $1}')

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

#export LOGLEVEL=INFO
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

echo "Job started at:"
date

export TORCHELASTIC_TIMEOUT=6000

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

time srun torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 1 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:$MASTER_PORT" \
    --rdzv-conf=timeout=6000 \
    quijote/train.py 

echo "Job finished at:"
date
