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
from multiprocessing import Pool


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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
    Ndevices = torch.cuda.device_count()

    norm_delta = 100,
    norm_vel = 1000,
    BoxSize = 25.
    grid = 8
    grid_sbox = 32
    npart_test = 128**3
    nMax_h = 20
    nvocab = 64
    nrand_sel_box = 64
    Mstar_cut = 8
    subsamp_ds = 2



    # dtype = 'float16'
    device_id = rank % torch.cuda.device_count()
    sdir = '/mnt/home/spandey/ceph/GOTHAM/data/camels'
    savefname = f'{sdir}/SPLIT_DATA_{Ndevices}_gpus_nspersim_subhalo_density3Dgrid_{grid_sbox}_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_nvocab{nvocab}_lgMmin_{Mstar_cut}.h5'
    # if rank == 0: print(f"Reading data from {savefname}", flush=True)
    with h5.File(savefname, 'r') as f:
        x_train_gpu = torch.tensor(f[f'x_train_dev_{rank}'][:]).to(torch.long).to(device_id, non_blocking=True)
        y_train_gpu = torch.tensor(f[f'y_train_dev_{rank}'][:]).to(torch.long).to(device_id, non_blocking=True)
        mask_train_gpu = torch.tensor(f[f'mask_train_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)
        params_train_gpu = torch.tensor(f[f'params_train_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)
        dm_train_gpu = torch.tensor(f[f'dm_train_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)

        x_val_gpu = torch.tensor(f[f'x_val_dev_{rank}'][:]).to(torch.long).to(device_id, non_blocking=True)
        y_val_gpu = torch.tensor(f[f'y_val_dev_{rank}'][:]).to(torch.long).to(device_id, non_blocking=True)
        mask_val_gpu = torch.tensor(f[f'mask_val_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)
        params_val_gpu = torch.tensor(f[f'params_val_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)
        dm_val_gpu = torch.tensor(f[f'dm_val_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)

        nvocab_total = f['nvocab_total'][()]
        grid_size = int(f['grid'][()])
        start_token = f['start_token'][()]
        pad_token = int(f['pad_token'][()])
        end_token = f['end_token'][()]
        max_sentence_length = f['max_sentence_length'][()]  
    f.close()

    # if rank == 0: print(f"Transferred data to GPU", flush=True)        

    # max_iters = 3000
    eval_interval = 10
    learning_rate = 5e-4
    eval_iters = 8
    n_embd = 96
    # n_head = 8
    # n_layer = 8

    n_head = 6
    n_layer = 6

    dropout = 0.2
    nparams = 6 # number of parameters in camels to append to the CNN features output
    vocab_size = nvocab_total
    block_size = max_sentence_length - 1
    print(f"block_size = {block_size}, vocab_size = {vocab_size}, pad_token = {pad_token}, max_sentence_length = {max_sentence_length}")
    print(f"nembd = {n_embd}, nhead = {n_head}, nlayer = {n_layer}, nparams = {nparams}, dropout = {dropout}")
    
    HaloConfig = {'block_size': block_size, 'vocab_size': vocab_size, 'n_layer': n_layer, 
                    'n_head': n_head, 'n_embd': n_embd, 'nparams': nparams, 'dropout': dropout, 
                    'bias': True, 'ksize': 3, 'density_grid_in': grid_size, 'density_grid_out': 4, 
                    'ninp_density': 30, 'pad_token': pad_token, 'flash': False}


    model = HaloDecoderModel(HaloConfig).to(device_id)

    # load the model checkpoint:
    cp_name = f'/mnt/home/spandey/ceph/GOTHAM/model_checkpoints/camels/model_encdec_ddp_PM_nvocab_64_nembed_64_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_RUN1.pt'
    checkpoint = torch.load(cp_name, map_location=f'cuda:{device_id}')    
    model.load_state_dict(checkpoint['model'])


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
            params = params_train_gpu

        elif split == 'val':
            x = x_val_gpu
            y = y_val_gpu
            mask = mask_val_gpu
            dm = dm_val_gpu
            params = params_val_gpu

        if batch_size is not None:
            x = x[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)
            y = y[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)
            mask = mask[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)
            dm = dm[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)
            params = params[batch_size*(ji):batch_size*(ji+1)].to(device_id, non_blocking=True)

        return x, y, mask, dm, params

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, MASK, DM, PARAMS = get_batch(split, batch_size = batch_size)
                with ctx:
                    logits, loss = model(X, DM, params=PARAMS, maskd=MASK, targets=Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out    

    decay_lr = True # whether to decay the learning rate
    decay_lr_model = 'cosine'
    warmup_iters = 400 # how many steps to warm up for
    lr_decay_iters = 1500 # should be ~= max_iters per Chinchilla
    min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
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
    running_mfu = -1.0    
    best_val_loss = 1e20
    # nbatches = 64
    batch_size = 320
    nbatches = len(x_train_gpu) // batch_size
    print(f"nbatches = {nbatches}, total train size = {len(x_train_gpu)}")
    max_iters = 6000
    eval_interval = 20
    save_separate_interval = 100

    # accumulation_steps = 1  # Accumulate gradients over 2 steps

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
                        torch.save(checkpoint, f'/mnt/home/spandey/ceph/GOTHAM/model_checkpoints/camels/model_encdec_ddp_PM_nvocab_64_nembed_64_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}.pt')                                 

                        if iter_num % save_separate_interval == 0 and (rank == 0):
                            torch.save(checkpoint, f'/mnt/home/spandey/ceph/GOTHAM/model_checkpoints/camels/model_encdec_ddp_PM_nvocab_64_nembed_64_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_iter_{iter_num}.pt')

        for ji in (range(nbatches)):
            model.require_backward_grad_sync = (ji == nbatches - 1)

            X, Y, MASK, DM, PARAMS = get_batch('train', ji, batch_size)
            with ctx:
                _, loss = model(X, DM, params=PARAMS, maskd=MASK, targets=Y)
            scaler.scale(loss).backward()   
            torch.cuda.empty_cache() 

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


    dist.destroy_process_group()

if __name__ == "__main__":
    train()