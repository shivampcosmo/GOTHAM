import dill
import numpy as np
import os
import sys

noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
ldir_stats = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/Pk_all_noise/'
pos = 'rsd'

saved_idx = np.loadtxt("/u/yzhang116/NN/inference/quijote_saved_idx.txt")

output_test = {}
output_train = {}

pk_train_truth = []
pk_train_mock = []
theta_train = []
pk_test_truth = []
pk_test_mock = []
theta_test = []
nbar_test_mock = []
nbar_test_truth = []
nbar_train_mock = []
nbar_train_truth = []

nhod_LH_samp = 10

for isim in range(2000):
    try:
        #saved_j = dill.load(open(ldir_stats + f'Pk_NGP_noise_{noise:.1f}_LH_' + str(isim) + '.dill', 'rb')) 
        saved_j = dill.load(open(ldir_stats + f'Pk_NGP_galnoise_{noise:.2f}_LH_' + str(isim) + '.dill', 'rb'))   
        #saved_w = dill.load(open(ldir_stats + 'WST_all/WST_galaxy_LH_' + str(isim) + '.dill', 'rb'))    
        if isim in saved_idx:
            for ihod in range(nhod_LH_samp):
                pk_test_mock.append(saved_j[pos + f'_Pk_mock_{ihod}'])
                pk_test_truth.append(saved_j[pos + f'_Pk_truth_{ihod}'])
                # saved_j[pos + f'_Bk_mock_0p08_{ihod}'][-1] might be negative, so we take [:-1] to exclude it
                #bk_mock = np.concatenate((saved_j[pos + f'_Bk_mock_0p08_{ihod}'], saved_j[pos + f'_Bk_mock_0p16_{ihod}'], saved_j[pos + f'_Bk_mock_0p32_{ihod}']))
                #bk_truth = np.concatenate((saved_j[pos + f'_Bk_truth_0p08_{ihod}'], saved_j[pos + f'_Bk_truth_0p16_{ihod}'], saved_j[pos + f'_Bk_truth_0p32_{ihod}']))

                #bk_all_mock.append(bk_mock)
                #bk_all_truth.append(bk_truth)
                theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
                theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())
                theta_comb = np.array(theta_cosmo + theta_hod)
                theta_test.append(theta_comb)
                #nbar_test_mock.append(saved_j[f'galsum_mock_{ihod}']['number density'])
                #nbar_test_truth.append(saved_j[f'galsum_truth_{ihod}']['number density'])
                nbar_test_mock.append(saved_j[f'galsum_mock_{ihod}']['number_density'])
                nbar_test_truth.append(saved_j[f'galsum_truth_{ihod}']['number_density'])
        else:
            for ihod in range(nhod_LH_samp):
                pk_train_mock.append(saved_j[pos + f'_Pk_mock_{ihod}'])
                pk_train_truth.append(saved_j[pos + f'_Pk_truth_{ihod}'])
                # saved_j[pos + f'_Bk_mock_0p08_{ihod}'][-1] might be negative, so we take [:-1] to exclude it
                #bk_mock = np.concatenate((saved_j[pos + f'_Bk_mock_0p08_{ihod}'], saved_j[pos + f'_Bk_mock_0p16_{ihod}'], saved_j[pos + f'_Bk_mock_0p32_{ihod}']))
                #bk_truth = np.concatenate((saved_j[pos + f'_Bk_truth_0p08_{ihod}'], saved_j[pos + f'_Bk_truth_0p16_{ihod}'], saved_j[pos + f'_Bk_truth_0p32_{ihod}']))

                #bk_all_mock.append(bk_mock)
                #bk_all_truth.append(bk_truth)
                theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
                theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())
                theta_comb = np.array(theta_cosmo + theta_hod)
                theta_train.append(theta_comb)
                #nbar_train_mock.append(saved_j[f'galsum_mock_{ihod}']['number density'])
                #nbar_train_truth.append(saved_j[f'galsum_truth_{ihod}']['number density'])
                nbar_train_mock.append(saved_j[f'galsum_mock_{ihod}']['number_density'])
                nbar_train_truth.append(saved_j[f'galsum_truth_{ihod}']['number_density'])
        
    except Exception as e:
        print(e)
        pass

pk_train_mock = np.array(pk_train_mock)
pk_train_truth = np.array(pk_train_truth)
pk_test_mock = np.array(pk_test_mock)
pk_test_truth = np.array(pk_test_truth)
theta_train = np.array(theta_train)
theta_test = np.array(theta_test)
nbar_test_mock = np.array(nbar_test_mock)
nbar_test_truth = np.array(nbar_test_truth)
nbar_train_mock = np.array(nbar_train_mock)
nbar_train_truth = np.array(nbar_train_truth)

output_train['pk_all_mock'] = pk_train_mock
output_train['pk_all_truth'] = pk_train_truth
output_train['k'] = saved_j['k_Pk_0']
output_train['theta_all'] = theta_train
output_train['nbar_mock'] = nbar_train_mock
output_train['nbar_truth'] = nbar_train_truth

output_test['pk_all_mock'] = pk_test_mock
output_test['pk_all_truth'] = pk_test_truth
output_test['k'] = saved_j['k_Pk_0']
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

save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/Pk_all_noise/Pk_nbar_NGP_galnoise_{noise:.2f}_200c_train.dill"
with open(save_path, 'wb') as f:
    dill.dump(output_train, f)

save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/Pk_all_noise/Pk_nbar_NGP_galnoise_{noise:.2f}_200c_test.dill"
with open(save_path, 'wb') as f:
    dill.dump(output_test, f)