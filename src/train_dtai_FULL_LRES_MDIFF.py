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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with key-value arguments')
    
    # Define arguments with their default values and types
    parser.add_argument('--grid_sbox', type=int, default=8, 
                        help='Grid size parameter')
    parser.add_argument('--add_space_token', type=lambda x: x.lower() == 'true', 
                        default=False, help='Whether to add space token')
    parser.add_argument('--subsel_type', type=str, default='all',
                        help='Subset selection type')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=400,
                        help='Maximum iterations')
    parser.add_argument('--n_embd', type=int, default=192,
                        help='Embedding dimension')                        
    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patch size for Vision Transformer')                                                
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        help='Maximum iterations')    
    parser.add_argument('--cnn_type', type=str, default='vit',
                        help='Maximum iterations')                            
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    grid_sbox = args.grid_sbox
    add_space_token = args.add_space_token
    subsel_type = args.subsel_type
    learning_rate = args.learning_rate
    max_iters = args.max_iters
    loss_type = args.loss_type
    cnn_type = args.cnn_type
    n_embd = args.n_embd
    patch_size = args.patch_size
    print(f"grid_sbox = {grid_sbox}, add_space_token = {add_space_token}, subsel_type = {subsel_type}, learning_rate = {learning_rate}, max_iters = {max_iters}, loss_type = {loss_type}, cnn_type = {cnn_type}")
    # print(f"grid_sbox = {grid_sbox}, add_space_token = {add_space_token}, subsel_type = {subsel_type}, learning_rate = {learning_rate}, max_iters = {max_iters}, loss_type = {loss_type}")

# try:
#     grid_sbox = int(ast.literal_eval(sys.argv[-5]))
#     add_space_token = bool(ast.literal_eval(sys.argv[-4]))
#     subsel_type = sys.argv[-3]
#     learning_rate = float(ast.literal_eval(sys.argv[-2]))
#     max_iters = int(ast.literal_eval(sys.argv[-1]))
# except:
#     grid_sbox = 8
#     add_space_token = False
#     subsel_type = 'all'    
#     learning_rate = 5e-4
#     max_iters = 250

# print(learning_rate, max_iters)

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
    # Ndevices = 16
    Ndevices = 24 
    # Ndevices = 2

    BoxSize = 1000.
    grid = 32
    # grid_sbox = 32
    nvocab = 64
    nrand_sel_box = 8192
    subsamp_ds = 1
    ds_fac_here = 1
    # ds_type_here = 'random'
    ds_type_here = 'seq'
    # rand_seed_dsfac = 0
    rand_seed_dsfac = 0
    # add_space_token = False
    # Mstar_cut = 8.5
    Mstar_cut = 12.7
    # Mstar_cut = 13.3
    DS_RES_POS_FAC = 4

    torch.cuda.empty_cache()
    device_id = rank % torch.cuda.device_count()
    sdir = '/work/hdd/bdne/spandey3/quijote_data/halo_gotham_data/process_split'
    savefname = f'{sdir}/SPLIT_DMO_DATA_{Ndevices}_gpus_density3Dgrid_{grid_sbox}_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}.h5'
    dist.barrier()
    with h5.File(savefname, 'r') as f:
        ind_all_train = np.arange(f[f'dm_train_dev_{rank}'][:].shape[0])
        ind_all_val = np.arange(f[f'dm_val_dev_{rank}'][:].shape[0])
        if ds_type_here == 'random':
            np.random.seed(rand_seed_dsfac)
            ind_all_train = np.random.permutation(ind_all_train)
            ind_all_val = np.random.permutation(ind_all_val)
            ind_sel_train = ind_all_train[::ds_fac_here]
            ind_sel_val = ind_all_val[::ds_fac_here]
        else:
            ind_sel_train = ind_all_train[rand_seed_dsfac::ds_fac_here]
            ind_sel_val = ind_all_val[rand_seed_dsfac::ds_fac_here]

        print(f"ind_sel_train = {ind_sel_train.shape}, ind_sel_val = {ind_sel_val.shape}", flush=True)
        dm_train_gpu = torch.tensor(f[f'dm_train_dev_{rank}'][:][ind_sel_train]).to(ptdtype).to(device_id, non_blocking=True)
        dm_val_gpu = torch.tensor(f[f'dm_val_dev_{rank}'][:][ind_sel_val]).to(ptdtype).to(device_id, non_blocking=True)
        grid_size = int(f['grid'][()])
    f.close()
    dist.barrier()
    torch.cuda.empty_cache()


    # savefname = f'{sdir}/SPLIT_HALO_DATA_{Ndevices}_gpus_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_nvocab{nvocab}_spacetoken_{add_space_token}_xMvc_{Mstar_cut}.h5'
    savefname = f'{sdir}/SPLIT_HALO_DATA_{Ndevices}_gpus_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_grid_{nvocab//DS_RES_POS_FAC}_nvocab{nvocab}_spacetoken_{add_space_token}_xMv1Dc_{Mstar_cut}.h5'
    # savefname = f'{sdir}/SPLIT_HALO_DATA_{Ndevices}_gpus_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_grid_{nvocab//DS_RES_POS_FAC}_nvocab{nvocab}_spacetoken_{add_space_token}_Mvcxyz_{Mstar_cut}.h5'
    dist.barrier()
    with h5.File(savefname, 'r') as f:
        x_train_gpu = torch.tensor(f[f'x_train_dev_{rank}'][:][ind_sel_train]).to(torch.long).to(device_id, non_blocking=True)
        y_train_gpu = torch.tensor(f[f'y_train_dev_{rank}'][:][ind_sel_train]).to(torch.long).to(device_id, non_blocking=True)
        mask_train_gpu = torch.tensor(f[f'mask_train_dev_{rank}'][:][ind_sel_train]).to(ptdtype).to(device_id, non_blocking=True)
        params_train_gpu = torch.tensor(f[f'params_train_dev_{rank}'][:][ind_sel_train]).to(ptdtype).to(device_id, non_blocking=True)

        x_val_gpu = torch.tensor(f[f'x_val_dev_{rank}'][:][ind_sel_val]).to(torch.long).to(device_id, non_blocking=True)
        y_val_gpu = torch.tensor(f[f'y_val_dev_{rank}'][:][ind_sel_val]).to(torch.long).to(device_id, non_blocking=True)
        mask_val_gpu = torch.tensor(f[f'mask_val_dev_{rank}'][:][ind_sel_val]).to(ptdtype).to(device_id, non_blocking=True)
        params_val_gpu = torch.tensor(f[f'params_val_dev_{rank}'][:][ind_sel_val]).to(ptdtype).to(device_id, non_blocking=True)

        nvocab_total = f['nvocab_total'][()]
        start_token = f['start_token'][()]
        pad_token = int(f['pad_token'][()])
        end_token = f['end_token'][()]
        max_sentence_length = f['max_sentence_length'][()]  
    f.close()
    dist.barrier()
    torch.cuda.empty_cache()

    indices = torch.arange(dm_train_gpu.shape[1])

    dm_train_gpu = dm_train_gpu[:,indices,...]
    dm_val_gpu = dm_val_gpu[:,indices,...]

    print(subsel_type, indices, dm_train_gpu.shape, dm_val_gpu.shape, add_space_token)
    
    # max_iters = 3000
    eval_interval = 10
    # learning_rate = 3e-4
    # max_iters = 1500
    eval_iters = 8
    # n_embd = 384
    # n_embd = 192
    # n_head = 8
    # n_layer = 8

    # n_head = 8
    # n_layer = 4

    n_head = 4
    # n_layer = 2   
    n_layer = 4       

    dropout = 0.2
    nparams = 5 # number of parameters in camels to append to the CNN features output
    vocab_size = nvocab_total
    block_size = max_sentence_length - 1
    print(f"block_size = {block_size}, vocab_size = {vocab_size}, pad_token = {pad_token}, max_sentence_length = {max_sentence_length}")
    print(f"nembd = {n_embd}, nhead = {n_head}, nlayer = {n_layer}, nparams = {nparams}, dropout = {dropout}")

    # if grid_sbox == 32:
    #     layers_types =  ['res', 'res', 'res', 'res']
    # if grid_sbox == 16:
    #     layers_types =  ['res', 'res', 'res']
    # if grid_sbox == 8:
    #     layers_types =  ['res','res']
    if grid_sbox == 32:
        layers_types =  ['res_cbam', 'res_cbam', 'res_cbam', 'res_cbam']
    if grid_sbox == 16:
        layers_types =  ['res_cbam', 'res_cbam', 'res_cbam']
    if grid_sbox == 8:
        layers_types =  ['res_cbam']    
    
    if cnn_type == 'res_cbam':
        dmo_cond_embed_type = 'resnet'
        layers_types =  ['res_cbam']
    elif cnn_type == 'vit':
        dmo_cond_embed_type = 'vit'
        layers_types =  ['cnn']
    elif cnn_type == 'vit_cbam':
        dmo_cond_embed_type = 'vit'
        layers_types =  ['res_cbam']        
    else:
        print(f"Invalid cnn_type: {cnn_type}")
        
    HaloConfig = {'block_size': block_size, 'vocab_size': vocab_size, 'n_layer': n_layer, 
                    'n_head': n_head, 'n_embd': n_embd, 'nparams': nparams, 'dropout': dropout, 
                    'bias': False, 'ksize': 3, 'density_grid_in': grid_size, 'density_grid_out': 4, 
                    'ninp_density': dm_train_gpu.shape[1], 'pad_token': pad_token, 'flash': True,
                    'dmo_cond_embed_type':dmo_cond_embed_type, 'layers_types':layers_types,
                    # 'n_layers_vit': 4, 'n_heads_vit': 8}
                    # 'n_layers_vit': n_layer, 'n_heads_vit': n_head}
                    'patch_size':patch_size,
                    # 'n_layers_vit': 2, 'n_heads_vit': 8}
                    'n_layers_vit': 2, 'n_heads_vit': 4}
                    # 'n_layers_vit': 3, 'n_heads_vit': 8}
                    # 'n_layers_vit': 4, 'n_heads_vit': 4}


    model = HaloDecoderModel(HaloConfig).to(device_id)

    # load the model checkpoint:
    
    # cp_name = f'/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FINAL4_fidfinetune_TEST_model_hres_encdec_ddp_grid_8_nvocab_64_nembed_256_nhead_8_nrandsubsel_8192_subselDMOfields_all_Mstarcut_12.7_spacetoken_False_maxiter_1500_lr_0.0005.pt'
    # cp_name = '/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FINAL3_TEST_model_hres_encdec_ddp_grid_8_nvocab_64_nembed_256_nhead_8_nrandsubsel_2048_subselDMOfields_all_Mstarcut_12.7_spacetoken_False_maxiter_1500_lr_0.0005.pt'
    # cp_name = '/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FINAL5_nofidfinetune_seq_TEST_model_hres_encdec_ddp_grid_8_nvocab_64_nembed_256_nhead_8_nrandsubsel_2048_subselDMOfields_all_Mstarcut_12.7_spacetoken_False_maxiter_1500_lr_0.0005.pt'

    # cp_name = '/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FINAL_TEST_model_hres_encdec_ddp_grid_8_nvocab_64_nembed_256_nhead_8_nrandsubsel_2048_subselDMOfields_all_Mstarcut_12.7_spacetoken_False_maxiter_1500_lr_0.0005.pt'
    # if n_embd == 192:
    # cp_name = '/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FULL_wposembed_vit/TESTv0_LRES_MDIFF_cnn_vit_cbam_ps_1_nrandsubsel_8192_cross_entropy_iter_260_seq_grid_8_nvocab_64_nembed_128_nhead_4_nrandsubsel_8192_subselDMOfields_all_Mstarcut_12.7_maxiter_400_lr_0.0002.pt'
    # cp_name = '/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FULL_wposembed_vit/TESTv0_LRES_MDIFF_cnn_vit_cbam_ps_1_nrandsubsel_8192_cross_entropy_iter_280_seq_grid_8_nvocab_64_nembed_128_nhead_4_nrandsubsel_8192_subselDMOfields_all_Mstarcut_12.7_maxiter_400_lr_0.0001.pt'    
    # checkpoint = torch.load(cp_name, map_location=f'cuda:{device_id}')    
    # model.load_state_dict(checkpoint['model'])

    # elif n_embd == 192:
    #     cp_name = '/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/iter_save/TEST2_LRES_MDIFF_cnn_vit_nrandsubsel_8192_cross_entropy_iter_196_seq_grid_8_nvocab_64_nembed_192_nhead_8_nrandsubsel_8192_subselDMOfields_all_Mstarcut_13.3_maxiter_400_lr_0.002.pt'

    # checkpoint = torch.load(cp_name, map_location=f'cuda:{device_id}')    
    # model.load_state_dict(checkpoint['model'])

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
    warmup_iters = 100 # how many steps to warm up for
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
    min_lr = learning_rate/10. # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
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
    # batch_size = 1024
    # batch_size = 4096
    batch_size = 2048 
    # batch_size = 4096
    # batch_size = 3500    
    # batch_size = 768
    nbatches = len(x_train_gpu) // batch_size
    print(f"nbatches = {nbatches}, total train size = {len(x_train_gpu)}")
    
    eval_interval = 4
    save_separate_interval = 10

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
                        # torch.save(checkpoint, f'/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FINAL5_afterfinetune_{ds_type_here}_TEST_model_hres_encdec_ddp_grid_{grid_sbox}_nvocab_{nvocab}_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/(subsamp_ds * ds_fac_here))}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_spacetoken_{add_space_token}_maxiter_{max_iters}_lr_{learning_rate}.pt')                                 
                        # torch.save(checkpoint, f'/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FINAL6_nofidfinetune_{ds_type_here}_TEST_model_hres_encdec_ddp_grid_{grid_sbox}_nvocab_{nvocab}_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/(subsamp_ds * ds_fac_here))}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_spacetoken_{add_space_token}_maxiter_{max_iters}_lr_{learning_rate}.pt')                                 
                        # torch.save(checkpoint, f'/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/iter_save/FINAL2_iter_{iter_num}_{ds_type_here}_TEST_model_hres_encdec_ddp_grid_{grid_sbox}_nvocab_{nvocab}_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/(subsamp_ds * ds_fac_here))}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_spacetoken_{add_space_token}_maxiter_{max_iters}_lr_{learning_rate}.pt')                                 
                        # torch.save(checkpoint, f'/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/iter_save/TESTLOSS_{loss_type}_CNNTYPE_{cnn_type}_iter_{iter_num}_{ds_type_here}_TEST_model_hres_encdec_ddp_grid_{grid_sbox}_nvocab_{nvocab}_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/(subsamp_ds * ds_fac_here))}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_spacetoken_{add_space_token}_maxiter_{max_iters}_lr_{learning_rate}.pt')                                 
                        # torch.save(checkpoint, f'/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FULL_wposembed_vit/TESTv1_LRES_MDIFF_cnn_{cnn_type}_ps_{patch_size}_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_{loss_type}_iter_{iter_num}_{ds_type_here}_grid_{grid_sbox}_nvocab_{nvocab}_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/(subsamp_ds * ds_fac_here))}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_maxiter_{max_iters}_lr_{learning_rate}.pt')                                 
                        torch.save(checkpoint, f'/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FULL_wposembed_vit/TEST_xMv1Dc_LRES_MDIFF_cnn_{cnn_type}_ps_{patch_size}_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_{loss_type}_iter_{iter_num}_{ds_type_here}_grid_{grid_sbox}_nvocab_{nvocab}_nembed_{n_embd}_nhead_{n_head}_nrandsubsel_{int(nrand_sel_box/(subsamp_ds * ds_fac_here))}_subselDMOfields_{subsel_type}_Mstarcut_{Mstar_cut}_maxiter_{max_iters}_lr_{learning_rate}.pt')                                 

        for ji in (range(nbatches)):
            model.require_backward_grad_sync = (ji == nbatches - 1)

            X, Y, MASK, DM, PARAMS = get_batch('train', ji, batch_size)
            with ctx:
                _, loss = model(X, DM, params=PARAMS, maskd=MASK, targets=Y, loss_type=loss_type)
            scaler.scale(loss).backward()   
            torch.cuda.empty_cache() 

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


    dist.destroy_process_group()

if __name__ == "__main__":
    train()