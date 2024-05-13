import sys, os
import numpy as np
import torch
dev = torch.device("cuda")
import torch.optim as optim
root_dir = '/mnt/home/spandey/ceph/CHARFORMER/'
os.chdir(root_dir)
# import colossus
import pickle as pk
# append the root_dir to the path
sys.path.append(root_dir)
from src.model_enc_dec import *
import matplotlib.pyplot as pl
pl.rc('text', usetex=True)
# Palatino
pl.rc('font', family='DejaVu Sans')
import numpy as np
import h5py as h5


import torch.distributed as dist

f = h5.File('/mnt/home/spandey/ceph/CHARFORMER/data/df_halo_part_ngp_xyzM_tokenized_density3Dgrid_32_isim_012_snap_3.h5', 'r')
dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all = f['dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all'][:]
delta_box_all_squeezed_all = f['delta_box_all_squeezed_all'][:]
f.close()    



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(rank, world_size, batch_size=32, pin_memory=False):

    # data_dir = '/mnt/home/spandey/ceph/CHARFORMER/data/'
    # df = pk.load(open('/mnt/home/spandey/ceph/CHARFORMER/data/df_halo_part_ngp_xyzM_tokenized_density3Dgrid_32_isim_0_snap_3.pkl','rb'))
    # dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed = np.memmap(os.path.join(data_dir, 'dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_isim_0_to_2.bin'), dtype=np.uint32, mode='r')
    # delta_box_all_squeezed = np.memmap(os.path.join(data_dir, 'delta_box_squeezed_all_isim_0_to_2.bin'), dtype=np.float32, mode='r')

    # dataset = Your_Dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

# saved = {'dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed':dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed,
#         'dfhalo_ngp_wxyzM': dfhalo_ngp_wxyzM,
#         'Nhalos_truth': Nhalos_truth,
#         'delta_box_all_squeezed': delta_box_all_squeezed}
# dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed = df['dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed']
# dfhalo_ngp_wxyzM = df['dfhalo_ngp_wxyzM']
# Nhalos_truth = df['Nhalos_truth']
# delta_box_all_squeezed = df['delta_box_all_squeezed']

# delta_box_all_squeezed = np.moveaxis(delta_box_all_squeezed, -1, 1)
# print(delta_box_all_squeezed.shape)

        
n1 = int(0.8*len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed)) # first 90% will be train, rest val
n2 = int(1.0*len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed)) # first 90% will be train, rest val
train_data_halos = dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed[:n1]
val_data_halos = dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed[n1:n2]

train_data_dm = delta_box_all_squeezed[:n1]
val_data_dm = delta_box_all_squeezed[n1:n2]

        
# train_data_dm.shape, val_data_dm.shape
# dfhalo_ngp_wxyzM_train = dfhalo_ngp_wxyzM[:n1]
# dfhalo_ngp_wxyzM_val = dfhalo_ngp_wxyzM[n1:n2]

# Nhalos_truth_train = Nhalos_truth[:n1]
# Nhalos_truth_val = Nhalos_truth[n1:n2]

# data = train_data if split == 'train' else val_data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.tensor(train_data_halos[:, :-1])
y = torch.tensor(train_data_halos[:, 1:])
dm_train = torch.tensor(train_data_dm).bfloat16()

mask_train = x != 1
mask_batch_reshape = torch.tile(mask_train.unsqueeze(-1), (1, 1, mask_train.shape[1]))
mask_batch_reshape_transpose = mask_batch_reshape.transpose(1, 2)
# mask_train = torch.logical_not(mask_batch_reshape & mask_batch_reshape_transpose)
mask_train = torch.logical_not(mask_batch_reshape_transpose)
# mask_train = torch.logical_not(mask_batch_reshape)
masked_logits = torch.zeros(mask_train.shape)
mask_train = masked_logits.masked_fill(mask_train, float('-inf')).bfloat16()

# x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)
x_train = x.long()
y_train = y.long()

# dm_train = dm.to(device).bfloat16()
# mask_train = torch.tensor(mask_train).to(device).bfloat16()



x = torch.tensor(val_data_halos[:, :-1])
y = torch.tensor(val_data_halos[:, 1:])
dm_val = torch.tensor(val_data_dm).bfloat16()

mask_val = x != 1
mask_batch_reshape = torch.tile(mask_val.unsqueeze(-1), (1, 1, mask_val.shape[1]))
mask_batch_reshape_transpose = mask_batch_reshape.transpose(1, 2)
# mask_val = torch.logical_not(mask_batch_reshape & mask_batch_reshape_transpose)
mask_val = torch.logical_not(mask_batch_reshape_transpose)
# mask_val = torch.logical_not(mask_batch_reshape)
masked_logits = torch.zeros(mask_val.shape)
mask_val = masked_logits.masked_fill(mask_val, float('-inf')).bfloat16()
# x, y = torch.tensor(x).to(device), torch.tensor(y).to(device)
x_val = x.long()
y_val = y.long()
# dm_val = dm.to(device).bfloat16()
# mask_val = torch.tensor(mask_val).to(device).bfloat16()


# start_token = 0
# pad_token = 1
# end_token = int(torch.max(x_train).cpu().numpy()) - 1
# space_token = int(torch.max(x_train).cpu().numpy())
# max_sentence_length = 1 + nMax_h*4 + 1 + (nMax_h-1)


nvocab = 128
bins_digitize = np.linspace(-1e-3, 1, nvocab)
# bins_digitize.insert(0, -1)
bins_digitize = np.insert(bins_digitize, 0, -1)
nMax_h = 32
start_token = 0
pad_token = 1
end_token = nvocab + 1
space_token = nvocab + 2
max_sentence_length = 1 + nMax_h*4 + 1 + (nMax_h-1)


import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
# hyperparameters
# batch_size = 16 # how many independent sequences will we process in parallel?
# block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 48
n_head = 4
n_layer = 4
dropout = 0.2
# ------------
# vocab_size = int(torch.max(x_train).cpu().numpy()) + 1
vocab_size = nvocab + 3
block_size = x_train.shape[-1]
batch_size = 1024

# n_embd_dm = 64
# n_head_dm = 4
# n_layer_dm = 3
# dropout_dm = 0.2
# vocab_size_dm = 3
# block_size_dm = dm_train.shape[1]
# batch_size = 2048


@dataclass
class HaloConfig:
    block_size: int = block_size
    vocab_size: int = vocab_size
    n_layer: int = n_layer
    n_head: int = n_head
    n_embd: int = n_embd
    dropout: float = dropout
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    ksize : int = 3
    density_grid_in : int = 32
    density_grid_out : int = 4
    ninp_density : int = 3

    start_token : int = start_token
    pad_token : int = pad_token
    end_token : int = end_token
    space_token : int = space_token