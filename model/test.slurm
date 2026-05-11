#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=generate_reynolds
#SBATCH --partition=ghx4

#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=/u/yzhang116/logs/%x.o%j
#SBATCH --error=/u/yzhang116/logs/%x.o%j


module purge             
module load cudatoolkit/25.5_12.9

# <<< conda initialize <<<
source ~/.bashrc
conda activate nn_test

which python
which nvcc
which gcc
echo $CC

export CC=gcc
export CXX=g++

which python
which nvcc
which gcc
echo $CC

#srun hostname
#python quijote/generate_catalog_CV.py
python quijote/generate_reynolds.py
#python quijote/save_embed_logits_block.py
#python quijote/generate_catalog_permuted.py

#srun python quijote/generate.py 1837664 16800 checkpoint_res 2e-05

#srun python quijote/generate.py 1869064 3450 checkpoint_cos 5e-06
#srun python quijote/generate.py 1869064 4000 checkpoint_cos 5e-06
#srun python quijote/generate.py 1869064 5200 checkpoint_cos 5e-06

#srun python quijote/generate.py 1872234 2450 checkpoint_cos 1e-05

#srun python quijote/generate.py 1872236 350 checkpoint_cos 1e-05
#srun python quijote/generate.py 1872236 3950 checkpoint_cos 1e-05
#srun python quijote/generate.py 1872236 6500 checkpoint_cos 1e-05
#srun python quijote/generate.py 1872236 7050 checkpoint_cos 1e-05

#srun python quijote/generate.py 1872237 250 checkpoint_cos 1e-05

#srun python quijote/generate.py 1868970 150 checkpoint_cos 5e-06
#srun python quijote/generate.py 1868970 1000 checkpoint_cos 5e-06