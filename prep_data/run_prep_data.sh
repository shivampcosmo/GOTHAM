#!/bin/bash
#SBATCH --account=bdne-delta-cpu
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=32
#SBATCH --time=0-03:30            # time (DD-HH:MM)
#SBATCH --job-name=test_prep_data
#SBATCH --output=/projects/bdne/spandey3/GOTHAM/prep_data/logs/%x.%j.out
#SBATCH --error=/projects/bdne/spandey3/GOTHAM/prep_data/logs/%x.%j.err
#SBATCH --exclusive --mem=0

# export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
module load python
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/external/python/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/external/python/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/sw/external/python/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/sw/external/python/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate /projects/bdne/spandey3/envs/charm
time srun --export=ALL python /projects/bdne/spandey3/GOTHAM/prep_data/prep_data_LHset_camels.py
echo "done"