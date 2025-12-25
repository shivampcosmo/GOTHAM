import numpy as np
import os, sys

# seed = int(sys.argv[1])
# print(seed)

# sim_id = int(sys.argv[2])
# print(sim_id)

js_min = 0
js_max = 10
js_all = np.arange(js_min, js_max).astype(int)

for js in js_all:

    ldir = '/mnt/home/fvillaescusa/ceph/Quijote/Snapshots/latin_hypercube_HR/' + str(js) + '/ICs'
    ffile_orig = open(ldir + '/2LPT.param', 'r')
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


    print(values_all)
    ffile = open('./2LPT.param', 'r')
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

        # if 'Seed' in line:
        #     line = 'Seed \t \t %d'%seed
        
        g += line
            
    # print(g)
    os.makedirs('./%d/'%js, exist_ok=True)
    ffile = open('./%d/2LPT.param'%js, 'w')
    ffile.write(g)
    ffile.close()




# Omega            0.3175    % Total matter density  (at z=0)
# OmegaLambda      0.6825    % Cosmological constant (at z=0)
# OmegaBaryon      0.0000    % Baryon density        (at z=0)
# OmegaDM_2ndSpecies  0.0    % Omega for a second dark matter species (at z=0)
# HubbleParam      0.6711    % Hubble paramater (may be used for power spec parameterization)

# Redshift         0       % Starting redshift
# Sigma8         0.834       % power spectrum normalization at z=0