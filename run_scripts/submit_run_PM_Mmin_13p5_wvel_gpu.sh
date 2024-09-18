#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=12:00:00
#SBATCH --job-name=lowLR_wvel_orig_NV64_512p3_Mmin_13p5
#SBATCH -p gpu
#SBATCH -C a100-80gb&ib-a100
#SBATCH --mem=1000G
#SBATCH --gpus-per-node=4
#SBATCH --output=/mnt/home/spandey/ceph/CHARFORMER/src/slurm_logs/PM/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARFORMER/src/slurm_logs/PM/%x.%j.err

# export MASTER_PORT=12367
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi

master_node=$SLURMD_NODENAME

cd /mnt/home/spandey/ceph/GOTHAM/src/
srun python `which torchrun` \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $SLURM_GPUS_PER_NODE \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $master_node:29500 \
        test_ddp_PM_Mmin_13p5_nv64_wvel.py
echo "done"
