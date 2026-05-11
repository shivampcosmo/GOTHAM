import dill
import numpy as np
import os
import sys

noise = 0.7
ldir_stats = '/work/hdd/bdne/yzhang116/quijote_galaxy/WST_all/'
pos = 'rsd'

saved_idx = np.loadtxt("/u/yzhang116/NN/inference/quijote_saved_idx.txt")

output_test = {}
output_train = {}

pk_train_truth = []
pk_train_mock = []
pk2_train_truth = []
pk2_train_mock = []
theta_train = []
pk_test_truth = []
pk_test_mock = []
pk2_test_truth = []
pk2_test_mock = []
theta_test = []
nbar_test_mock = []
nbar_test_truth = []
nbar_train_mock = []
nbar_train_truth = []

nhod_LH_samp = 10
grid1 = 128
grid2 = 192
#model = 'J6Q2'
model = 'angular'

for isim in range(2000):
    try:
        saved_j = dill.load(open(ldir_stats + f'WST_{model}_NGP_galnoise_{noise:.1f}_galaxy_LH_{isim}.dill', 'rb'))    
        #saved_j = dill.load(open(ldir_stats + f'WST_{model}_galaxy_LH_{isim}.dill', 'rb'))
        #saved_j = dill.load(open(ldir_stats + f'WST_{model}_NGP_noise_{noise:.1f}_galaxy_LH_{isim}.dill', 'rb'))
        #saved_w = dill.load(open(ldir_stats + 'WST_all/WST_galaxy_LH_' + str(isim) + '.dill', 'rb'))    
        if isim in saved_idx:
            for ihod in range(nhod_LH_samp):                
                pk_test_mock.append(np.concatenate([saved_j[f'rsd_s0_mock_{ihod}_{grid1:d}'], saved_j[f'rsd_s1_mock_{ihod}_{grid1:d}'], saved_j[f'rsd_s2_mock_{ihod}_{grid1:d}']]))
                pk_test_truth.append(np.concatenate([saved_j[f'rsd_s0_truth_{ihod}_{grid1:d}'], saved_j[f'rsd_s1_truth_{ihod}_{grid1:d}'], saved_j[f'rsd_s2_truth_{ihod}_{grid1:d}']]))
                pk2_test_mock.append(np.concatenate([saved_j[f'rsd_s0_mock_{ihod}_{grid2:d}'], saved_j[f'rsd_s1_mock_{ihod}_{grid2:d}'], saved_j[f'rsd_s2_mock_{ihod}_{grid2:d}']]))
                pk2_test_truth.append(np.concatenate([saved_j[f'rsd_s0_truth_{ihod}_{grid2:d}'], saved_j[f'rsd_s1_truth_{ihod}_{grid2:d}'], saved_j[f'rsd_s2_truth_{ihod}_{grid2:d}']]))
                # saved_j[pos + f'_Bk_mock_0p08_{ihod}'][-1] might be negative, so we take [:-1] to exclude it
                #bk_mock = np.concatenate((saved_j[pos + f'_Bk_mock_0p08_{ihod}'], saved_j[pos + f'_Bk_mock_0p16_{ihod}'], saved_j[pos + f'_Bk_mock_0p32_{ihod}']))
                #bk_truth = np.concatenate((saved_j[pos + f'_Bk_truth_0p08_{ihod}'], saved_j[pos + f'_Bk_truth_0p16_{ihod}'], saved_j[pos + f'_Bk_truth_0p32_{ihod}']))

                #bk_all_mock.append(bk_mock)
                #bk_all_truth.append(bk_truth)
                theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
                theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())
                theta_comb = np.array(theta_cosmo + theta_hod)
                theta_test.append(theta_comb)
                nbar_test_mock.append(saved_j[f'galsum_mock_{ihod}']['number density'])
                nbar_test_truth.append(saved_j[f'galsum_truth_{ihod}']['number density'])
        else:
            for ihod in range(nhod_LH_samp):
                pk_train_mock.append(np.concatenate([saved_j[f'rsd_s0_mock_{ihod}_{grid1:d}'], saved_j[f'rsd_s1_mock_{ihod}_{grid1:d}'], saved_j[f'rsd_s2_mock_{ihod}_{grid1:d}']]))
                pk_train_truth.append(np.concatenate([saved_j[f'rsd_s0_truth_{ihod}_{grid1:d}'], saved_j[f'rsd_s1_truth_{ihod}_{grid1:d}'], saved_j[f'rsd_s2_truth_{ihod}_{grid1:d}']]))
                pk2_train_mock.append(np.concatenate([saved_j[f'rsd_s0_mock_{ihod}_{grid2:d}'], saved_j[f'rsd_s1_mock_{ihod}_{grid2:d}'], saved_j[f'rsd_s2_mock_{ihod}_{grid2:d}']]))
                pk2_train_truth.append(np.concatenate([saved_j[f'rsd_s0_truth_{ihod}_{grid2:d}'], saved_j[f'rsd_s1_truth_{ihod}_{grid2:d}'], saved_j[f'rsd_s2_truth_{ihod}_{grid2:d}']]))
                # saved_j[pos + f'_Bk_mock_0p08_{ihod}'][-1] might be negative, so we take [:-1] to exclude it
                #bk_mock = np.concatenate((saved_j[pos + f'_Bk_mock_0p08_{ihod}'], saved_j[pos + f'_Bk_mock_0p16_{ihod}'], saved_j[pos + f'_Bk_mock_0p32_{ihod}']))
                #bk_truth = np.concatenate((saved_j[pos + f'_Bk_truth_0p08_{ihod}'], saved_j[pos + f'_Bk_truth_0p16_{ihod}'], saved_j[pos + f'_Bk_truth_0p32_{ihod}']))

                #bk_all_mock.append(bk_mock)
                #bk_all_truth.append(bk_truth)
                theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
                theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())
                theta_comb = np.array(theta_cosmo + theta_hod)
                theta_train.append(theta_comb)
                nbar_train_mock.append(saved_j[f'galsum_mock_{ihod}']['number density'])
                nbar_train_truth.append(saved_j[f'galsum_truth_{ihod}']['number density'])
        
    except Exception as e:
        print(f"Error processing file for isim={isim}: {e}", flush=True)
        pass

pk_train_mock = np.array(pk_train_mock)
pk_train_truth = np.array(pk_train_truth)
pk_test_mock = np.array(pk_test_mock)
pk_test_truth = np.array(pk_test_truth)
pk2_train_mock = np.array(pk2_train_mock)
pk2_train_truth = np.array(pk2_train_truth)
pk2_test_mock = np.array(pk2_test_mock)
pk2_test_truth = np.array(pk2_test_truth)
theta_train = np.array(theta_train)
theta_test = np.array(theta_test)
nbar_test_mock = np.array(nbar_test_mock)
nbar_test_truth = np.array(nbar_test_truth)
nbar_train_mock = np.array(nbar_train_mock)
nbar_train_truth = np.array(nbar_train_truth)

output_train[f'wst_mock_{grid1:d}_all'] = pk_train_mock
output_train[f'wst_truth_{grid1:d}_all'] = pk_train_truth
output_train[f'wst_mock_{grid2:d}_all'] = pk2_train_mock
output_train[f'wst_truth_{grid2:d}_all'] = pk2_train_truth
output_train['theta_all'] = theta_train
output_train['nbar_mock'] = nbar_train_mock
output_train['nbar_truth'] = nbar_train_truth

output_test[f'wst_mock_{grid1:d}_all'] = pk_test_mock
output_test[f'wst_truth_{grid1:d}_all'] = pk_test_truth
output_test[f'wst_mock_{grid2:d}_all'] = pk2_test_mock
output_test[f'wst_truth_{grid2:d}_all'] = pk2_test_truth
output_test['theta_all'] = theta_test
output_test['nbar_mock'] = nbar_test_mock
output_test['nbar_truth'] = nbar_test_truth

print("data shapes:")
print("pk_train_mock:", pk_train_mock.shape)
print("pk_train_truth:", pk_train_truth.shape)
print("theta_train:", theta_train.shape)
print("pk_test_mock:", pk_test_mock.shape)
print("pk_test_truth:", pk_test_truth.shape)
print("theta_test:", theta_test.shape)  
print("nbar_test_mock:", nbar_test_mock.shape)
print("nbar_test_truth:", nbar_test_truth.shape)
print("nbar_train_mock:", nbar_train_mock.shape)
print("nbar_train_truth:", nbar_train_truth.shape)

save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/WST_all/WST_nbar_{model}_NGP_galnoise_{noise:.1f}_train.dill"
#save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/WST_all/WST_nbar_{model}_NGP_noise_0.0_train.dill"
with open(save_path, 'wb') as f:
    dill.dump(output_train, f)

save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/WST_all/WST_nbar_{model}_NGP_galnoise_{noise:.1f}_test.dill"
#save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/WST_all/WST_nbar_{model}_NGP_noise_0.0_test.dill"
with open(save_path, 'wb') as f:
    dill.dump(output_test, f)