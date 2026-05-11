import numpy as np
import os, sys
from tqdm import tqdm
# seed = int(sys.argv[1])
# print(seed)

# sim_id = int(sys.argv[2])
# print(sim_id)

js_min = 663
js_max = 664
js_all = np.arange(js_min, js_max).astype(int)

sdir_base = '/mnt/ceph/users/spandey/discodj_runs/test_2gpc/LH'

for js in tqdm(js_all):

    # check if the directory exists, else create it:
    sdir = sdir_base + '/%d/'%js
    sdir_IC = sdir_base + '/%d/ICs'%js
    os.makedirs(sdir, exist_ok=True)
    os.makedirs(sdir_IC, exist_ok=True)

    ldir = '/mnt/home/fvillaescusa/ceph/Quijote/Snapshots/latin_hypercube_HR/' + str(js)
    Pkmm_z0_file_orig = ldir + '/ICs/Pk_mm_z=0.000.txt'
    os.system('cp %s %s'%(Pkmm_z0_file_orig, sdir_IC + '/Pk_mm_z=0.000.txt'))

    cosmo_params_orig = ldir + '/Cosmo_params.dat'
    os.system('cp %s %s'%(cosmo_params_orig, sdir_IC + '/Cosmo_params.dat'))

    ffile_orig = open(ldir + '/ICs/2LPT.param', 'r')
    f = ffile_orig.readlines()
    ffile_orig.close()

    variables_all = ['Seed', 'Omega ', 'OmegaLambda ', 'OmegaBaryon ', 'OmegaDM_2ndSpecies ', 'HubbleParam ', 'Redshift', 'Sigma8']
    values_all = {}
    g = ''   
    for variable in variables_all:
        for line in f:        
            if variable in line:
                # print(line.split())
                if variable == 'Seed':
                    value = int(line.split()[1])
                else:
                    value = float(line.split()[1])
                values_all[variable] = value

    ffile = open('./2LPT_base_1024.param', 'r')
    f = ffile.readlines()
    ffile.close()

    g = ''   
    for line in f:
        for variable in variables_all:
            if variable in line:
                # print(line.split())
                value = values_all[variable]
                line = line.replace(line.split()[1], str(value))
                # print(line)

        g += line
            
    ffile = open(sdir_IC + '/2LPT_1024.param', 'w')
    ffile.write(g)
    ffile.close()
