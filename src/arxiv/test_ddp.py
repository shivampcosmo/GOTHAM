import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
# dev = torch.device("cuda")
import torch.optim as optim
root_dir = '/mnt/home/spandey/ceph/CHARFORMER/'
os.chdir(root_dir)
# import colossus
import pickle as pk
# append the root_dir to the path
sys.path.append(root_dir)
from src.model_enc_dec import *
import numpy as np
import h5py as h5
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from contextlib import nullcontext
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# dtype = 'float32'
# if master_process:
    # os.makedirs(out_dir, exist_ok=True)
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP

import os



def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class CustomDataset(Dataset):
    def __init__(self, X_halo, X_DM, MASK_halo, Y_pred):
        self.X1 = X_halo
        self.X2 = X_DM
        self.MASK_X1 = MASK_halo        
        self.Y = Y_pred

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, index):
        return self.X1[index], self.X2[index], self.MASK_X1[index], self.Y[index]

    def __iter__(self):
        start = self.rank * (len(self) // self.world_size)
        end = start + (len(self) // self.world_size)
        return iter(range(start, end))

    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def set_epoch(self, epoch):
        # Optionally shuffle your data for each epoch
        pass

def get_data_split(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed, delta_box_all_squeezed, n1_fac=0.8, n2_fac=1.0):
    n1 = int(n1_fac*len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed)) # first 90% will be train, rest val
    n2 = int(n2_fac*len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed)) # first 90% will be train, rest val
    train_data_halos = dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed[:n1]
    val_data_halos = dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed[n1:n2]

    train_data_dm = delta_box_all_squeezed[:n1]
    val_data_dm = delta_box_all_squeezed[n1:n2]

    x = torch.tensor(train_data_halos[:, :-1])
    y = torch.tensor(train_data_halos[:, 1:])
    dm = torch.tensor(train_data_dm)
    mask_train_orig = x != 1
    mask_train = torch.logical_not(mask_train_orig)
    masked_logits = torch.zeros(mask_train.shape)
    mask_train_final = masked_logits.masked_fill(mask_train, float('-inf'))
    mask_train = mask_train_final[:,None,:]
    x, y = torch.tensor(x), torch.tensor(y)
    x_train = x.long()
    y_train = y.long()
    dm_train = dm.bfloat16()
    mask_train = torch.tensor(mask_train).bfloat16()

    x = torch.tensor(val_data_halos[:, :-1])
    y = torch.tensor(val_data_halos[:, 1:])
    dm = torch.tensor(val_data_dm)
    mask_val_orig = x != 1
    mask_val = torch.logical_not(mask_val_orig)
    masked_logits = torch.zeros(mask_val.shape)
    mask_val_final = masked_logits.masked_fill(mask_val, float('-inf'))
    mask_val = mask_val_final[:,None,:]
    x, y = torch.tensor(x), torch.tensor(y)
    x_val = x.long()
    y_val = y.long()
    dm_val = dm.bfloat16()
    mask_val = torch.tensor(mask_val).bfloat16()

    return x_train, y_train, dm_train, mask_train, x_val, y_val, dm_val, mask_val





from dataclasses import dataclass
# if __name__ == '__main__':
    # hyperparameters
    # batch_size = 16 # how many independent sequences will we process in parallel?
    # block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 10
learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 48
n_head = 4
n_layer = 4
dropout = 0.2
# ------------
vocab_size = 131
block_size = 161
batch_size = 1950

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

    pad_token : int = 1



def demo_basic():

    device = 'cuda'
    compile = True # use PyTorch 2.0 to compile the model to be faster
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda'
    dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    dtype = 'bfloat16'
    if rank == 0: print(f"Loading data", flush=True)    
    f = h5.File('/mnt/home/spandey/ceph/CHARFORMER/data/df_halo_part_ngp_xyzM_tokenized_density3Dgrid_32_isim_012_snap_3.h5', 'r')
    dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all = f['dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all'][:]
    delta_box_all_squeezed_all = f['delta_box_all_squeezed_all'][:]
    f.close()
    if rank == 0: print(f"Loaded data", flush=True)        

    x_train, y_train, dm_train, mask_train, x_val, y_val, dm_val, mask_val = get_data_split(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all, delta_box_all_squeezed_all, 0.8, 1.0)
    if rank == 0: print(f"Got split with sizes {x_train.shape} and {x_val.shape}", flush=True)        

    labels = torch.randn(20, 5).to(device_id)
    if rank == 0: print(f"Transferred labels to GPU", flush=True)      

    start = rank * (len(x_train) // torch.cuda.device_count())
    end = start + (len(x_train) // torch.cuda.device_count())
    # x_train_gpu = (x_train[start:end,...]).to(device_id, non_blocking=True)
    # dm_train_gpu = (dm_train[start:end,...]).to(device_id, non_blocking=True)
    # mask_train_gpu = (mask_train[start:end,...]).to(device_id, non_blocking=True)
    # y_train_gpu = (y_train[start:end,...]).to(device_id, non_blocking=True)

    x_train_gpu = (x_train[start:end,...]).to(device_id)
    dm_train_gpu = (dm_train[start:end,...]).to(device_id)
    mask_train_gpu = (mask_train[start:end,...]).to(device_id)
    y_train_gpu = (y_train[start:end,...]).to(device_id)
    print(f"I am rank {rank} and will process train data from {start} to {end}.")
    if rank == 0: print(f"Transferred train data to GPU", flush=True)        

    start = rank * (len(x_val) // torch.cuda.device_count())
    end = start + (len(x_val) // torch.cuda.device_count())
    x_val_gpu = (x_val[start:end,...]).to(device_id, non_blocking=True)
    dm_val_gpu = (dm_val[start:end,...]).to(device_id, non_blocking=True)
    mask_val_gpu = (mask_val[start:end,...]).to(device_id, non_blocking=True)
    y_val_gpu = (y_val[start:end,...]).to(device_id, non_blocking=True)
    print(f"I am rank {rank} and will process val data from {start} to {end}.")    
    if rank == 0: print(f"Transferred test data to GPU", flush=True)        

    model = HaloDecoderModel(HaloConfig).to(device_id)
    # if rank == 0: 
    checkpoint = torch.load('/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec_testddp.pt', map_location=f'cuda:{device_id}')
    model.load_state_dict(checkpoint['model'])
    if rank == 0: print(f"Init model and loaded to GPU", flush=True)            
    # model.to(device).bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # unoptimized_model = model
    # model = torch.compile(model) # requires PyTorch 2.0    
    # model.load_state_dict(torch.load('/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec.pt'))

    model = DDP(model, device_ids=[device_id])

    def get_batch(split, ji=0, batch_size=None):
        if split == 'train':
            x = x_train_gpu
            y = y_train_gpu
            mask = mask_train_gpu
            dm = dm_train_gpu

        elif split == 'val':
            x = x_val_gpu
            y = y_val_gpu
            mask = mask_val_gpu
            dm = dm_val_gpu

        if batch_size is not None:
            x = x[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)
            y = y[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)
            mask = mask[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)
            dm = dm[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)

        return x, y, mask, dm
    
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, MASK, DM = get_batch(split, batch_size = batch_size)
                with ctx:
                    logits, loss = model(X, DM, maskd=MASK, targets=Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out    

    decay_lr = True # whether to decay the learning rate
    warmup_iters = 300 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    iter_num = 0
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module
    running_mfu = -1.0    
    best_val_loss = 1e20
    nbatches = 10
    max_iters = 5000
    eval_interval = 20
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 and (rank == 0):
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': HaloConfig,
                        }
                        print(f"saving checkpoint")
                        torch.save(checkpoint, '/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec_testddp.pt')            

        for ji in (range(nbatches)):
            model.require_backward_grad_sync = (ji == nbatches - 1)

            # print(f"I am rank {rank} and will process batch data from {batch_size*ji} to {batch_size*(ji+1)}.")                
            X, Y, MASK, DM = get_batch('train', ji, batch_size)
            with ctx:
                _, loss = model(X, DM, maskd=MASK, targets=Y)
                # loss = loss
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()   

        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        if (iter_num % 10) == 0 and (rank == 0):
            print(f"iter {iter_num}, loss: {loss.item()}")                 

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


    # model = ToyModel().to(device_id)
    # ddp_model = DDP(model, device_ids=[device_id])

    # loss_fn = nn.MSELoss()
    # optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # optimizer.zero_grad()
    # outputs = ddp_model(torch.randn(20, 10))
    # labels = torch.randn(20, 5).to(device_id)
    # loss_fn(outputs, labels).backward()
    # optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()