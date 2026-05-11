import dill
import numpy as np
import os
import sys

# Note for Yao:
# when apply extended HOD, noise is :.2f, and noise0 version is galnoise_0.00
# and the key is 'number_density' instead of 'number density'

noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
ldir_stats = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/Bk_all_noise/'
pos = 'rsd'

saved_idx = np.loadtxt("/u/yzhang116/NN/inference/quijote_saved_idx.txt", dtype=int)

output_test = {}
output_train = {}

pk_train_truth = []
pk_train_mock = []
qk_train_truth = []
qk_train_mock = []
theta_train = []
pk_test_truth = []
pk_test_mock = []
qk_test_truth = []
qk_test_mock = []
theta_test = []
nbar_test_mock = []
nbar_test_truth = []
nbar_train_mock = []
nbar_train_truth = []

nhod_LH_samp = 10

for isim in range(2000):
    try:
        saved_j = dill.load(open(ldir_stats + f'Bk_NGP_galnoise_{noise:.2f}_LH_' + str(isim) + '.dill', 'rb'))    
        #saved_w = dill.load(open(ldir_stats + 'WST_all/WST_galaxy_LH_' + str(isim) + '.dill', 'rb'))    
        if isim in saved_idx:
            for ihod in range(nhod_LH_samp):
                pk_test_mock.append(np.concatenate([saved_j[pos + f'_Bk_mock_0p08_{ihod}'], saved_j[pos + f'_Bk_mock_0p16_{ihod}'], saved_j[pos + f'_Bk_mock_0p32_{ihod}']]))
                pk_test_truth.append(np.concatenate([saved_j[pos + f'_Bk_truth_0p08_{ihod}'], saved_j[pos + f'_Bk_truth_0p16_{ihod}'], saved_j[pos + f'_Bk_truth_0p32_{ihod}']]))
                qk_test_mock.append(np.concatenate([saved_j[pos + f'_Qk_mock_0p08_{ihod}'], saved_j[pos + f'_Qk_mock_0p16_{ihod}'], saved_j[pos + f'_Qk_mock_0p32_{ihod}']]))
                qk_test_truth.append(np.concatenate([saved_j[pos + f'_Qk_truth_0p08_{ihod}'], saved_j[pos + f'_Qk_truth_0p16_{ihod}'], saved_j[pos + f'_Qk_truth_0p32_{ihod}']]))
                # saved_j[pos + f'_Bk_mock_0p08_{ihod}'][-1] might be negative, so we take [:-1] to exclude it
                #bk_mock = np.concatenate((saved_j[pos + f'_Bk_mock_0p08_{ihod}'], saved_j[pos + f'_Bk_mock_0p16_{ihod}'], saved_j[pos + f'_Bk_mock_0p32_{ihod}']))
                #bk_truth = np.concatenate((saved_j[pos + f'_Bk_truth_0p08_{ihod}'], saved_j[pos + f'_Bk_truth_0p16_{ihod}'], saved_j[pos + f'_Bk_truth_0p32_{ihod}']))

                #bk_all_mock.append(bk_mock)
                #bk_all_truth.append(bk_truth)
                theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
                theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())
                theta_comb = np.array(theta_cosmo + theta_hod)
                theta_test.append(theta_comb)
        else:
            for ihod in range(nhod_LH_samp):
                pk_train_mock.append(np.concatenate([saved_j[pos + f'_Bk_mock_0p08_{ihod}'], saved_j[pos + f'_Bk_mock_0p16_{ihod}'], saved_j[pos + f'_Bk_mock_0p32_{ihod}']]))
                pk_train_truth.append(np.concatenate([saved_j[pos + f'_Bk_truth_0p08_{ihod}'], saved_j[pos + f'_Bk_truth_0p16_{ihod}'], saved_j[pos + f'_Bk_truth_0p32_{ihod}']]))
                qk_train_mock.append(np.concatenate([saved_j[pos + f'_Qk_mock_0p08_{ihod}'], saved_j[pos + f'_Qk_mock_0p16_{ihod}'], saved_j[pos + f'_Qk_mock_0p32_{ihod}']]))
                qk_train_truth.append(np.concatenate([saved_j[pos + f'_Qk_truth_0p08_{ihod}'], saved_j[pos + f'_Qk_truth_0p16_{ihod}'], saved_j[pos + f'_Qk_truth_0p32_{ihod}']]))
                # saved_j[pos + f'_Bk_mock_0p08_{ihod}'][-1] might be negative, so we take [:-1] to exclude it
                #bk_mock = np.concatenate((saved_j[pos + f'_Bk_mock_0p08_{ihod}'], saved_j[pos + f'_Bk_mock_0p16_{ihod}'], saved_j[pos + f'_Bk_mock_0p32_{ihod}']))
                #bk_truth = np.concatenate((saved_j[pos + f'_Bk_truth_0p08_{ihod}'], saved_j[pos + f'_Bk_truth_0p16_{ihod}'], saved_j[pos + f'_Bk_truth_0p32_{ihod}']))

                #bk_all_mock.append(bk_mock)
                #bk_all_truth.append(bk_truth)
                theta_hod = list(saved_j[f'theta_hod_{ihod}'].values())
                theta_cosmo = list(saved_j[f'theta_cosmo_{ihod}'].values())
                theta_comb = np.array(theta_cosmo + theta_hod)
                theta_train.append(theta_comb)
        
    except Exception as e:
        print(e)
        pass

pk_train_mock = np.array(pk_train_mock)
pk_train_truth = np.array(pk_train_truth)
pk_test_mock = np.array(pk_test_mock)
pk_test_truth = np.array(pk_test_truth)
theta_train = np.array(theta_train)
theta_test = np.array(theta_test)
qk_train_mock = np.array(qk_train_mock)
qk_train_truth = np.array(qk_train_truth)
qk_test_mock = np.array(qk_test_mock)
qk_test_truth = np.array(qk_test_truth)


output_train['bk_all_mock'] = pk_train_mock
output_train['bk_all_truth'] = pk_train_truth
output_train['qk_all_mock'] = qk_train_mock
output_train['qk_all_truth'] = qk_train_truth
output_train['theta_all'] = theta_train

output_test['bk_all_mock'] = pk_test_mock
output_test['bk_all_truth'] = pk_test_truth
output_test['qk_all_mock'] = qk_test_mock
output_test['qk_all_truth'] = qk_test_truth
output_test['theta_all'] = theta_test


print("data shapes:")
print("bk_train_mock:", pk_train_mock.shape)
print("bk_train_truth:", pk_train_truth.shape)
print("theta_train:", theta_train.shape)
print("bk_test_mock:", pk_test_mock.shape)
print("bk_test_truth:", pk_test_truth.shape)
print("theta_test:", theta_test.shape)  
print("qk_train_mock:", qk_train_mock.shape)
print("qk_train_truth:", qk_train_truth.shape)
print("qk_test_mock:", qk_test_mock.shape)
print("qk_test_truth:", qk_test_truth.shape)


save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/Bk_all_noise/Bk_NGP_galnoise_{noise:.2f}_200c_train.dill"
with open(save_path, 'wb') as f:
    dill.dump(output_train, f)

save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext/Bk_all_noise/Bk_NGP_galnoise_{noise:.2f}_200c_test.dill"
with open(save_path, 'wb') as f:
    dill.dump(output_test, f)

exit()

import dill
import numpy as np
import matplotlib.pyplot as plt
noise = 0.8

save_path = f"/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod/Bk_all_noise/Bk_NGP_noise_{noise:.2f}_test.dill"
with open(save_path, 'rb') as f:
   data =  dill.load(f)

for i in range(30):
    plt.plot(data['qk_all_mock'][i]/data['qk_all_truth'][i])
plt.hlines(1, 0, 24, colors='k', linestyles='dashed')
plt.savefig("test.png")