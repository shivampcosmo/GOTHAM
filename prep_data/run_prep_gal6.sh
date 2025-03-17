#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH -C genoa
#SBATCH --job-name=TEST1
#SBATCH -p ccm
#SBATCH --output=/mnt/home/spandey/ceph/halo_gotham/GOTHAM/prep_data/logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/halo_gotham/GOTHAM/prep_data/logs/%x.%j.err

# __conda_setup="$('/mnt/home/spandey/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/mnt/home/spandey/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/mnt/home/spandey/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/mnt/home/spandey/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup

source ~/miniconda3/bin/activate nbodykit


cd /mnt/home/spandey/ceph/halo_gotham/GOTHAM/prep_data/

time srun python /mnt/home/spandey/ceph/halo_gotham/GOTHAM/prep_data/process_halo_props.py 400 1600
echo "done"