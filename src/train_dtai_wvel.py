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
import ast

subsel_type = sys.argv[-1] if len(sys.argv) > 1 else "all"
try:
     # if len(sys.argv) > 2:
    add_space_token = bool(ast.literal_eval(sys.argv[-2]))
except:
    add_space_token = False

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
    # Ndevices = torch.cuda.device_count()
    Ndevices = 8

    BoxSize = 25.
    grid = 8
    grid_sbox = 32
    nvocab = 64
    nrand_sel_box = 128
    subsamp_ds = 1
    # add_space_token = False
    # Mstar_cut = 8.5
    Mstar_cut = 9.0    

    device_id = rank % torch.cuda.device_count()
    sdir = '/work/hdd/bdne/spandey3/camels_tng/gotham_data/process_split'
    savefname = f'{sdir}/SPLIT_DMO_DATA_{Ndevices}_gpus_density3Dgrid_{grid_sbox}_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}.h5'
    dist.barrier()
    with h5.File(savefname, 'r') as f:
        dm_train_gpu = torch.tensor(f[f'dm_train_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)
        dm_val_gpu = torch.tensor(f[f'dm_val_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)
        grid_size = int(f['grid'][()])
    f.close()
    dist.barrier()
    savefname = f'{sdir}/SPLIT_GALAXY_DATA_{Ndevices}_gpus_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_nvocab{nvocab}_spacetoken_{add_space_token}_wSDSS_photometry_gri_velx_Mstarcut_{Mstar_cut}.h5'
    # if add_space_token:
    #     savefname = f'{sdir}/SPLIT_GALAXY_DATA_{Ndevices}_gpus_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_nvocab{nvocab}_spacetoken_{add_space_token}_wSDSS_photometry_gri_velx_Mstarcut_{Mstar_cut}.h5'
    # else:
    #     savefname = f'{sdir}/SPLIT_GALAXY_DATA_{Ndevices}_gpus_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_nvocab{nvocab}_wSDSS_photometry_gri_velx_Mstarcut_{Mstar_cut}.h5'
    dist.barrier()
    with h5.File(savefname, 'r') as f:
        x_train_gpu = torch.tensor(f[f'x_train_dev_{rank}'][:]).to(torch.long).to(device_id, non_blocking=True)
        y_train_gpu = torch.tensor(f[f'y_train_dev_{rank}'][:]).to(torch.long).to(device_id, non_blocking=True)
        mask_train_gpu = torch.tensor(f[f'mask_train_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)
        params_train_gpu = torch.tensor(f[f'params_train_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)

        x_val_gpu = torch.tensor(f[f'x_val_dev_{rank}'][:]).to(torch.long).to(device_id, non_blocking=True)
        y_val_gpu = torch.tensor(f[f'y_val_dev_{rank}'][:]).to(torch.long).to(device_id, non_blocking=True)
        mask_val_gpu = torch.tensor(f[f'mask_val_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)
        params_val_gpu = torch.tensor(f[f'params_val_dev_{rank}'][:]).to(ptdtype).to(device_id, non_blocking=True)

        nvocab_total = f['nvocab_total'][()]
        start_token = f['start_token'][()]
        pad_token = int(f['pad_token'][()])
        end_token = f['end_token'][()]
        max_sentence_length = f['max_sentence_length'][()]  
    f.close()
    dist.barrier()
    
    if subsel_type == 'no_highz':
        indices = torch.arange(6)
    elif subsel_type == 'no_highz_no_vel':
        indices = torch.arange(3)
    elif subsel_type == 'no_highz_no_env':
        indices = torch.from_numpy(np.array([0,3,4,5]))
    elif subsel_type == 'no_vel':
        indices = torch.cat([torch.arange(i, i + 3) for i in range(0, 30, 6)])
    elif subsel_type == 'no_env':        
        indices1 = torch.cat([torch.arange(i+3, i + 6) for i in range(0, 30, 6)])
        indices2 = torch.cat([torch.arange(i, i + 1) for i in range(0, 30, 6)])
        indices, _ = torch.sort(torch.cat([indices1, indices2]))
    else:
        indices = torch.arange(dm_train_gpu.shape[1])

    dm_train_gpu = dm_train_gpu[:,indices,...]
    dm_val_gpu = dm_val_gpu[:,indices,...]

    print(subsel_type, indices, dm_train_gpu.shape, dm_val_gpu.shape, add_space_token)
    
    # max_iters = 3000
    eval_interval = 10
    learning_rate = 3e-4
    eval_iters = 8
    n_embd = 256
    # n_head = 8
    # n_layer = 8

    n_head = 8
    n_layer = 8

    dropout = 0.2
    nparams = 6 # number of parameters in camels to append to the CNN features output
    vocab_size = nvocab_total
    block_size = max_sentence_length - 1
    print(f"block_size = {block_size}, vocab_size = {vocab_size}, pad_token = {pad_token}, max_sentence_length = {max_sentence_length}")
    print(f"nembd = {n_embd}, nhead = {n_head}, nlayer = {n_layer}, nparams = {nparams}, dropout = {dropout}")
    
    HaloConfig = {'block_size': block_size, 'vocab_size': vocab_size, 'n_layer': n_layer, 
                    'n_head': n_head, 'n_embd': n_embd, 'nparams': nparams, 'dropout': dropout, 
                    'bias': True, 'ksize': 3, 'density_grid_in': grid_size, 'density_grid_out': 4, 
                    'ninp_density': dm_train_gpu.shape[1], 'pad_token': pad_token, 'flash': False}


    model = HaloDecoderModel(HaloConfig).to(device_id)

    # load the model checkpoint:
    
    cp_name = f'/projects/bdne/spandey3/GOTHAM/model_checkpoints/camels_photo_velx/model_hres_encdec_ddp_PM_nvocab_64_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_spacetoken_{add_space_token}.pt'
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
    lr_decay_iters = 2000 # should be ~= max_iters per Chinchilla
    min_lr = 3e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
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
    # batch_size = 320
    batch_size = 320
    nbatches = len(x_train_gpu) // batch_size
    print(f"nbatches = {nbatches}, total train size = {len(x_train_gpu)}")
    max_iters = 3000
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
                        torch.save(checkpoint, f'/projects/bdne/spandey3/GOTHAM/model_checkpoints/camels_photo_velx/model_hres_encdec_ddp_PM_nvocab_64_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_spacetoken_{add_space_token}.pt')                                 

                        if iter_num % save_separate_interval == 0 and (rank == 0):
                            torch.save(checkpoint, f'/projects/bdne/spandey3/GOTHAM/model_checkpoints/camels_photo_velx/model_hres_encdec_ddp_PM_nvocab_64_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_iter_{iter_num}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_spacetoken_{add_space_token}.pt')

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