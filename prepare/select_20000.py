import os
import numpy as np
import shutil
import pickle as pk


#halo_src_dir = "/work/hdd/bdne/yzhang116/halo_sentences_36/"    # 原始 halo 文件夹
#dm_src_dir = "/work/hdd/bdne/yzhang116/dmo_fields_4096_256/"            # 原始 density field 文件夹
halo_src_dir = '/work/hdd/bdne/spandey3/quijote_LH_discodj/halos_story_nsel_32768'
dm_src_dir = '/work/hdd/bdne/spandey3/quijote_LH_discodj/rhog_LH_np_512_nsnap_3_nsel_32768'

tot = 2000
indices = np.arange(tot)
np.random.seed(2026)
np.random.shuffle(indices)
train_idx = indices[:1800]
val_idx = indices[1800:1900]
test_idx = indices[1900:]

'''
np.savetxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_train_idx.txt', train_idx, fmt='%d')
np.savetxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_val_idx.txt', val_idx, fmt='%d')
np.savetxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_test_idx.txt', test_idx, fmt='%d')
'''
np.savetxt('/work/hdd/bdne/yzhang116/quijote_select_20000/quijote_train_idx.txt', train_idx, fmt='%d')
np.savetxt('/work/hdd/bdne/yzhang116/quijote_select_20000/quijote_val_idx.txt', val_idx, fmt='%d')
np.savetxt('/work/hdd/bdne/yzhang116/quijote_select_20000/quijote_test_idx.txt', test_idx, fmt='%d')

num_per_file = 20000
num_select = 20000
totnum = 32768
shuffle_num = 100
save_num = (num_select * shuffle_num) // num_per_file 

halo_train_dir = '/work/hdd/bdne/yzhang116/quijote_select_20000/quijote_halos'
dm_train_dir = '/work/hdd/bdne/yzhang116/quijote_select_20000/quijote_fields'

meta_f = pk.load(open('/work/hdd/bdne/spandey3/quijote_LH_discodj/halos_story_nsel_32768/sentence_params.pkl','rb'))
start_token = meta_f['start_token']
pad_token = int(meta_f['pad_token'])
end_token = meta_f['end_token']


temp = np.load(halo_src_dir + '/0/halo_sentence_LH_0.npy')
halo_save = np.zeros((num_select*shuffle_num, temp.shape[1]+1), dtype=temp.dtype)
temp = np.load(dm_src_dir + '/0/dmo_fields_subvols_grid_8_LH_0.npy')
dm_save = np.zeros((num_select*shuffle_num, *temp.shape[1:]), dtype=temp.dtype)

flag = 0
save_id = 0

for read_id in train_idx:
    print(f"Processing SimID: {read_id}")
    halo_data = np.load(halo_src_dir + f'/{read_id}/halo_sentence_LH_{read_id}.npy')
    dm_data = np.load(dm_src_dir + f'/{read_id}/dmo_fields_subvols_grid_8_LH_{read_id}.npy')
    sample = np.random.choice(totnum, size=num_select, replace=False)
    halo_temp = halo_data[sample]
    end_token_index = np.argmax(halo_temp == end_token, axis=1)
    n_halos = (end_token_index-6) // 8
    halo_save[flag*num_select:(flag+1)*num_select,:6] = halo_temp[:, :6]
    halo_save[flag*num_select:(flag+1)*num_select,6] = n_halos
    halo_save[flag*num_select:(flag+1)*num_select,7:] = halo_temp[:, 6:]
    dm_save[flag*num_select:(flag+1)*num_select] = dm_data[sample]
    flag += 1
    if flag == shuffle_num:
        save_index = np.arange(shuffle_num * num_select)
        np.random.shuffle(save_index)
        for i in range(save_num):
            selected_indices = save_index[i*num_per_file:(i+1)*num_per_file]
            np.save(os.path.join(halo_train_dir, f'halos_{save_id}.npy'), halo_save[selected_indices])
            np.save(os.path.join(dm_train_dir, f'fields_{save_id}.npy'), dm_save[selected_indices])
            save_id += 1
        flag = 0
'''
# for validation, no need to shuffle
temp = np.load(halo_src_dir + '/0/halo_sentence_LH_0.npy')
halo_save = np.zeros((num_select,temp.shape[1]+1), dtype=temp.dtype)

for read_id in val_idx:
    halo_data = np.load(halo_src_dir + f'/{read_id}/halo_sentence_LH_{read_id}.npy')
    dm_data = np.load(dm_src_dir + f'/{read_id}/dmo_fields_subvols_grid_8_LH_{read_id}.npy')
    sample = np.random.choice(totnum, size=num_select, replace=False)
    halo_temp = halo_data[sample]
    end_token_index = np.argmax(halo_temp == end_token, axis=1)
    n_halos = (end_token_index-6) // 8
    halo_save[:,:6] = halo_temp[:, :6]
    halo_save[:,6] = n_halos
    halo_save[:,7:] = halo_temp[:, 6:]
    np.save(os.path.join(halo_val_dir, f'halos_{read_id}.npy'), halo_save)
    np.save(os.path.join(dm_val_dir, f'fields_{read_id}.npy'), dm_data[sample])
'''