#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=32
#SBATCH --time=0-1:00            # time (DD-HH:MM)
#SBATCH --job-name=prep_gal_data
#SBATCH --partition=ghx4
#SBATCH --mem=512G
#SBATCH --gpus-per-node=1
#SBATCH --output=/projects/bdne/spandey3/GOTHAM/prep_data/logs/%x.%j.out
#SBATCH --error=/projects/bdne/spandey3/GOTHAM/prep_data/logs/%x.%j.err
#SBATCH --exclusive --mem=0

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


time srun --export=ALL python /projects/bdne/spandey3/GOTHAM/prep_data/process_gal_props.py
echo "done"