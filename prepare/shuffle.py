import os
import numpy as np
import shutil


halo_src_dir = "/work/hdd/bdne/yzhang116/halo_sentences_36/"    # 原始 halo 文件夹
dm_src_dir = "/work/hdd/bdne/yzhang116/dmo_fields_4096_256/"            # 原始 density field 文件夹

halo_train_dir = halo_src_dir+"train/"
halo_val_dir = halo_src_dir+"validation/"
halo_test_dir = halo_src_dir+"test/"

dm_train_dir = dm_src_dir+"train/"
dm_val_dir = dm_src_dir+"validation/"
dm_test_dir = dm_src_dir+"test/"


for d in [halo_train_dir, halo_val_dir, halo_test_dir, dm_train_dir, dm_val_dir, dm_test_dir]:
    os.makedirs(d, exist_ok=True)


halo_files = sorted([f for f in os.listdir(halo_src_dir) if f.endswith('.npy')],
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
dm_files = sorted([f for f in os.listdir(dm_src_dir) if f.endswith('.npy')],
                  key=lambda x: int(x.split('_')[-1].split('.')[0]))

# assert len(halo_files) == len(dm_files), "Halo files and DM fields must have the same number of files."

num_files = len(dm_files)
print(f"Total files: {num_files}")

np.random.seed(2025)
indices = np.arange(num_files)
np.random.shuffle(indices)


train_idx = indices[:800]
val_idx = indices[800:900]
test_idx = indices[900:]


def move_files(idx_list, halo_dest, dm_dest):
    for idx in idx_list:
        # shutil.move(os.path.join(halo_src_dir,halo_files[idx]), os.path.join(halo_dest, halo_files[idx]))
        shutil.move(os.path.join(dm_src_dir,dm_files[idx]), os.path.join(dm_dest, dm_files[idx]))

move_files(train_idx, halo_train_dir, dm_train_dir)
move_files(val_idx, halo_val_dir, dm_val_dir)
move_files(test_idx, halo_test_dir, dm_test_dir)

print("Files have been successfully split into train/val/test sets.")
