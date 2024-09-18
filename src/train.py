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
import torch.optim as optim
import pickle as pk
from model_enc_dec import *
import numpy as np
import h5py as h5
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from contextlib import nullcontext
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
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
    n1 = int(n1_fac*len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed)) 
    n2 = int(n2_fac*len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed)) 
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


f = h5.File('/mnt/home/spandey/ceph/GOTHAM/data/PM/df_halo_part_ngp_xyzM_tokenized_PM_384x384x384_density3Dgrid_32_isim_012_snap_3_nvocab64_Mmin_13p5.h5', 'r')
dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all = f['dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all'][:]
delta_box_all_squeezed_all = f['delta_box_all_squeezed_all'][:]
nvocab_total = f['nvocab_total'][()]
grid_size = f['grid'][()]
start_token = f['start_token'][()]
pad_token = f['pad_token'][()]
end_token = f['end_token'][()]
max_sentence_length = f['max_sentence_length'][()]
f.close()




from dataclasses import dataclass
max_iters = 6000
eval_interval = 10
learning_rate = 2e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2

vocab_size = nvocab_total
block_size = max_sentence_length - 1

batch_size = 1950


@dataclass
class HaloConfig:
    block_size: int = block_size
    vocab_size: int = vocab_size
    n_layer: int = n_layer
    n_head: int = n_head
    n_embd: int = n_embd
    dropout: float = dropout
    bias: bool = True 

    ksize : int = 3
    density_grid_in : int = grid_size
    density_grid_out : int = 4
    ninp_density : int = 3

    pad_token : int = pad_token
    flash : bool = False



def train():

    device = 'cuda'
    compile = True 
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    device_type = 'cuda'
    dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    device_id = rank % torch.cuda.device_count()

    dtype = 'bfloat16'

    x_train, y_train, dm_train, mask_train, x_val, y_val, dm_val, mask_val = get_data_split(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all, delta_box_all_squeezed_all, 0.8, 1.0)
    if rank == 0: print(f"Got split with sizes {x_train.shape} and {x_val.shape}", flush=True)        

    labels = torch.randn(20, 5).to(device_id)
    if rank == 0: print(f"Transferred labels to GPU", flush=True)      

    start = rank * (len(x_train) // torch.cuda.device_count())
    end = start + (len(x_train) // torch.cuda.device_count())

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
    if rank == 0: print(f"Init model and loaded to GPU", flush=True)            
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

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
    decay_lr_model = 'cosine'
    warmup_iters = 500 # how many steps to warm up for
    lr_decay_iters = 7500 # should be ~= max_iters per Chinchilla
    min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it, model='cosine'):
        # 1) linear warmup for warmup_iters steps
        if model == 'cosine':
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
        
        elif model == 'linear':
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            else:
                return learning_rate - (it - warmup_iters) * (learning_rate - min_lr) / (lr_decay_iters - warmup_iters)

        elif model == 'constant':
            return learning_rate

    iter_num = 0
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module
    running_mfu = -1.0    
    best_val_loss = 1e20
    nbatches = 10
    max_iters = 6000
    eval_interval = 20
    save_separate_interval = 400
    while True:
        lr = get_lr(iter_num, model=decay_lr_model) if decay_lr else learning_rate
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
                            'lr': lr
                        }
                        print(f"saving checkpoint")
                        torch.save(checkpoint, '/mnt/home/spandey/ceph/GOTHAM/model_checkpoints/model_encdec_ddp_PM_isim_012_nvocab_64_nembed_64_Mmin_13p5.pt')                                 

                        if iter_num % save_separate_interval == 0 and (rank == 0):
                            torch.save(checkpoint, f'/mnt/home/spandey/ceph/GOTHAM/model_checkpoints/model_encdec_ddp_PM_isim_012_nvocab_64_nembed_64_Mmin_13p5_{iter_num}.pt')

        for ji in (range(nbatches)):
            model.require_backward_grad_sync = (ji == nbatches - 1)

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


    dist.destroy_process_group()

if __name__ == "__main__":
    train()