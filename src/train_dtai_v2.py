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
from model_enc_dec_v2 import *
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
import os
import shutil
import numpy as np
import torch

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

def get_sim_number(filepath):
    filename = os.path.basename(filepath)
    number_part = filename.split('_')[-1]
    number = int(number_part.split('.')[0])
    return number

params_all = np.loadtxt('/projects/bdne/spandey3/GOTHAM/prep_data/camels_tng_LH_params.txt', usecols=range(1, 7))

class ExternalInputIterator:
    def __init__(self, input_dir, label_dir, batch_size, shard_id, num_shards, shuffle=False):
        files_in_inp_dir = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]
        self.dm_fields_files = sorted(files_in_inp_dir, key=get_sim_number)
        files_in_label_dir = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')]
        self.gal_prop_files = sorted(files_in_label_dir, key=get_sim_number)
        total = len(self.dm_fields_files)
        # print(total, len(self.gal_prop_files))
        assert total == len(self.gal_prop_files)
        
        # Shard the data: split in round-robin (recommended by DALI)
        self.indices = np.arange(total)[shard_id::num_shards]
        if shuffle:
            np.random.shuffle(self.indices)
        self.data_set_len = len(self.indices)
        self.batch_size = batch_size
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i >= self.data_set_len:
            self.i = 0
            raise StopIteration
        batch_inputs, batch_labels, batch_params = [], [], []
        for _ in range(self.batch_size):
            if self.i >= self.data_set_len:
                break
            idx = self.indices[self.i]
            batch_inputs.append(np.load(self.dm_fields_files[idx]))
            batch_labels.append(np.load(self.gal_prop_files[idx]))
            params = params_all[idx][None,:]
            # print(idx)
            nrep = batch_inputs[-1].shape[0]
            params_rep = np.repeat(params, nrep, axis=0)
            batch_params.append(params_rep)
            self.i += 1
        return (batch_inputs, batch_labels, batch_params)
    @property
    def size(self):
        return self.data_set_len

def create_dali_pipeline(input_dir, label_dir, batch_size, num_threads, device_id, shard_id, num_shards, shuffle=False):
    eii = ExternalInputIterator(input_dir, label_dir, batch_size, shard_id, num_shards, shuffle=shuffle)
    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    def npy_pipeline():
        dmo_fields, gal_tokens, params_rep = fn.external_source(source=eii, num_outputs=3)
        return dmo_fields.gpu(), gal_tokens.gpu(), params_rep.gpu()
    pipe = npy_pipeline()
    pipe.build()
    return pipe, eii.size


def parse_args():
    parser = argparse.ArgumentParser(description='Training script with key-value arguments')
    
    # Define arguments with their default values and types
    parser.add_argument('--grid_sbox', type=int, default=16, 
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
    parser.add_argument('--patch_size', type=int, default=2,
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

    # dist.init_process_group("nccl")
    print(int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", rank=int(os.environ.get("RANK", 0)), world_size=int(os.environ.get("WORLD_SIZE", 1)), device_id=torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'))
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # Ndevices = torch.cuda.device_count()
    Ndevices = 8
    # Ndevices = 2

    BoxSize = 25.
    grid = 8
    # grid_sbox = 32
    nvocab = 64
    nrand_sel_box = 512
    # subsamp_ds = 1
    # ds_fac_here = 1
    # ds_type_here = 'random'
    # ds_type_here = 'None'
    # rand_seed_dsfac = 0
    # rand_seed_dsfac = 1
    # add_space_token = False
    # Mstar_cut = 8.5
    # Mstar_cut = 12.7
    # Mstar_cut = 13.3
    # DS_RES_POS_FAC = 4

    nsnaps = 5

    Mstar_cut = 9.5
    DS_RES_POS_FAC = 1.0

    torch.cuda.empty_cache()
    device_id = rank % torch.cuda.device_count()

    meta_f = pk.load(open('/work/nvme/bdne/spandey3/camels_tng/gotham_data/LH/gal_props_ns512_3prop/metadata_galaxy_props_snap_90_grid_64_isim_0_nrandsubsel_512_nvocab64_spacetoken_False_wSDSS_photometry_velx_Mstarcut_9.5.pkl','rb'))
    # nvocab_total = meta_f['nvocab_total'][()]
    start_token = meta_f['start_token']
    pad_token = int(meta_f['pad_token'])
    end_token = meta_f['end_token']
    nvocab_total = end_token + 1
    max_sentence_length = meta_f['max_sentence_length']

    # input_dir, label_dir = args['input_dir'], args['label_dir']
    # batch_size = args['batch_size']
    # pipeline, n_samples_per_rank = create_dali_pipeline(
    #     input_dir, label_dir, batch_size, num_threads=4, device_id=device_id,
    #     shard_id=dist.get_rank(), num_shards=int(os.environ.get("WORLD_SIZE", 1)), shuffle=False)
    # dali_iterator = DALIGenericIterator(
    #     [pipeline],
    #     ['DM_fields', 'gal_tokens', 'params'],
    #     last_batch_policy=LastBatchPolicy.PARTIAL,
    #     auto_reset=True,
    # )

    n_head = 8
    n_layer = 4

    dropout = 0.2
    nparams = 6 # number of parameters in camels to append to the CNN features output
    vocab_size = nvocab_total
    block_size = max_sentence_length - 1
    if dist.get_rank() == 0:
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
        

    layers_types =  ['cnn']            
    HaloConfig = {'block_size': block_size, 'vocab_size': vocab_size, 'n_layer': n_layer, 
                    'n_head': n_head, 'n_embd': n_embd, 'nparams': nparams, 'dropout': dropout, 
                    'bias': False, 'ksize': 3, 'density_grid_in': grid_sbox, 'density_grid_out': 4, 
                    'ninp_density': 30, 'pad_token': pad_token, 'flash': True,
                    'dmo_cond_embed_type':dmo_cond_embed_type, 'layers_types':layers_types,
                    # 'n_layers_vit': 4, 'n_heads_vit': 8}
                    # 'n_layers_vit': n_layer, 'n_heads_vit': n_head}
                    'patch_size':patch_size,
                    'n_layers_vit': 4, 'n_heads_vit': 8}
                    # 'n_layers_vit': 3, 'n_heads_vit': 8}
                    # 'n_layers_vit': 4, 'n_heads_vit': 4}

    model = HaloDecoderModel(HaloConfig).to(device_id)
    # load the model checkpoint:

    # cp_name = f'/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FINAL4_fidfinetune_TEST_model_hres_encdec_ddp_grid_8_nvocab_64_nembed_256_nhead_8_nrandsubsel_8192_subselDMOfields_all_Mstarcut_12.7_spacetoken_False_maxiter_1500_lr_0.0005.pt'
    # checkpoint = torch.load(cp_name, map_location=f'cuda:{device_id}')    
    # model.load_state_dict(checkpoint['model'])

    if rank == 0: print(f"Init model and loaded to GPU", flush=True)            
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    def cosine_lr(step, total_steps, lr_min, lr_max):
        import math
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total_steps))

    # nepochs = max_iters
    best_loss = 1e20
    for jepoch in range(max_iters):
        if dist.get_rank() == 0:
            print(jepoch)
        input_dir = '/work/nvme/bdne/spandey3/camels_tng/gotham_data/LH/DMO_fields_ns512'
        label_dir = '/work/nvme/bdne/spandey3/camels_tng/gotham_data/LH/gal_props_ns512_3prop'
        batch_size = 10
        pipeline, n_samples_per_rank = create_dali_pipeline(
            input_dir, label_dir, batch_size, num_threads=4, device_id=device_id,
            shard_id=dist.get_rank(), num_shards=int(os.environ.get("WORLD_SIZE", 1)), shuffle=False)
        dali_iterator = DALIGenericIterator(
            [pipeline],
            ['DM_fields', 'gal_tokens', 'params'],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True
        )
        
        args = {'num_examples': 1000, 'lr_min': 1e-6, 'lr_max': learning_rate/((1 + jepoch)**0.75), 'batch_size': batch_size}
        # batch_size = 2
        global_batch_size = batch_size * 1
        total_batches = int(np.ceil(args['num_examples'] / global_batch_size))
        lr_min = args['lr_min']
        lr_max = args['lr_max']
        # Each process will see total_batches_per_rank batches (they all are equal or off by 1)
        batches_seen = 0
        
        sub_batch_size = 512
        nsub_batches = int(np.ceil(global_batch_size * 512 / sub_batch_size))
        # shuffle_sub_batches = False
        shuffle_sub_batches = True
        # print(nsub_batches)
        for step, batch in enumerate(dali_iterator):
            data_batch = batch[0]
            DM = torch.flatten(data_batch['DM_fields'], start_dim=0, end_dim=1)
            DM = torch.moveaxis(DM, -1, 1)  # move the last dimension to the second position

            gal_tokens = torch.flatten(data_batch['gal_tokens'], start_dim=0, end_dim=1)
            X = gal_tokens[:, :-1].to(torch.long)
            Y = gal_tokens[:, 1:].to(torch.long)
            mask = torch.logical_not(X != pad_token)
            masked_logits = torch.zeros(mask.shape, device=X.device, dtype=torch.float32)
            MASK = masked_logits.masked_fill(mask, float('-inf'))[:,None,:]

            PARAMS = data_batch['params'].to(torch.float32)
            PARAMS = torch.flatten(PARAMS, start_dim=0, end_dim=1)

            global_step = step  # local across this rank, but all ranks have unique data; global batches is fine

            # Compute global batch for LR schedule (all GPUs see different batches)
            # For one-epoch run: global_step = (local_rank) * local_batches_sofar + step
            lr = cosine_lr(global_step, total_batches-1, lr_min, lr_max)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            loss_mean_here = 0
            loss_all_array = []
            all_inds = np.arange(X.shape[0])
            if shuffle_sub_batches:
                np.random.shuffle(all_inds)
            for js in range(nsub_batches-1):
                model.require_backward_grad_sync = (js == nsub_batches - 1)
                ind_js = all_inds[js*sub_batch_size:(js+1)*sub_batch_size]
                # if step >= 60:
                    # print(js, len(ind_js))
                try:
                    with ctx:
                        _, loss = model(X[ind_js], DM[ind_js], 
                                    params=PARAMS[ind_js], maskd=MASK[ind_js], 
                                    targets=Y[ind_js], loss_type=loss_type)
                    scaler.scale(loss).backward()   
                except:
                    pass
                loss_mean_here += loss.item()
                loss_all_array.append(loss.item())
            # loss_mean_here /= nsub_batches
            loss_all_array = np.array(loss_all_array)
            loss_max_here = np.max(loss_all_array)
            loss_min_here = np.min(loss_all_array)
            loss_mean_here = np.mean(loss_all_array)

            # optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # torch.cuda.empty_cache()

            # if dist.get_rank() == 0:
            # print(f"[GPU {0} step {global_step+1}/{total_batches}] LR: {lr:.6f} | Loss: {loss.item():.4f}")

            if dist.get_rank() == 0:
                if np.mod(step + 1, 10) == 0:
                    print(f"[GPU {0} step {global_step+1}/{total_batches}] LR: {lr:.6f} | loss mean: {loss_mean_here:.4f}, loss min: {loss_min_here:.4f}, loss max: {loss_max_here:.4f}")

            batches_seen += 1
            if batches_seen >= total_batches:
                break  # Make sure you go through all samples ONCE, globally

            # if loss is less than previous best, save the checkpoint:
            if dist.get_rank() == 0:
                if loss_mean_here < best_loss:
                    best_loss = loss_mean_here
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'global_step': global_step,
                        'loss': best_loss,
                    }
                    torch.save(checkpoint, f'/projects/bdne/spandey3/FINAL_GOTHAM/GOTHAM/src/tmp/v5_checkpoint_step_{step*(jepoch+1)}.pt')
                    print(f"Checkpoint saved at step {global_step} with loss mean {best_loss:.4f}, loss min: {loss_min_here:.4f}, loss max: {loss_max_here:.4f}")
        
        dist.barrier()

            # termination conditions
            # if iter_num > max_iters:
            #     break

    if dist.get_rank() == 0:
        print("Training finished on all ranks.")
    dist.destroy_process_group()

    # for step, batch in enumerate(dali_iterator):
    #     data_batch = batch[0]
    #     DM = torch.flatten(data_batch['DM_fields'], start_dim=0, end_dim=1)
    #     DM = torch.moveaxis(DM, -1, 1)  # move the last dimension to the second position

    #     gal_tokens = torch.flatten(data_batch['gal_tokens'], start_dim=0, end_dim=1)
    #     X = gal_tokens[:, :-1].to(torch.long)
    #     Y = gal_tokens[:, 1:].to(torch.long)
    #     mask = torch.logical_not(X != pad_token)
    #     masked_logits = torch.zeros(mask.shape, device=X.device, dtype=torch.float32)
    #     MASK = masked_logits.masked_fill(mask, float('-inf'))[:,None,:]

    #     global_step = step  # local across this rank, but all ranks have unique data; global batches is fine

    #     # Compute global batch for LR schedule (all GPUs see different batches)
    #     # For one-epoch run: global_step = (local_rank) * local_batches_sofar + step
    #     lr = cosine_lr(global_step, total_batches-1, lr_min, lr_max)
    #     for pg in optimizer.param_groups:
    #         pg['lr'] = lr

    #     with ctx:
    #         _, loss = model(X, DM, params=PARAMS, maskd=MASK, targets=Y, loss_type=loss_type)
    #     scaler.scale(loss).backward()   

    #     # optimizer.step()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #     scaler.step(optimizer)
    #     scaler.update()
    #     optimizer.zero_grad(set_to_none=True)
    #     # torch.cuda.empty_cache()

    #     if dist.get_rank() == 0:
    #         print(f"[GPU {local_rank} step {global_step+1}/{total_batches}] LR: {lr:.6f} | Loss: {loss.item():.4f}")

    #     batches_seen += 1
    #     if batches_seen >= total_batches:
    #         break  # Make sure you go through all samples ONCE, globally

    #     # termination conditions
    #     if iter_num > max_iters:
    #         break

    # dist.barrier()
    # if dist.get_rank() == 0:
    #     print("Training finished on all ranks.")
    # dist.destroy_process_group()

if __name__ == "__main__":
    train()