import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.optim as optim
import pickle as pk
from model_enc_dec import *
import numpy as np
from torch.nn import functional as F
from dataclasses import dataclass
from contextlib import nullcontext
from multiprocessing import Pool
import ast
import math
import gc
import argparse
import shutil
import wandb  # Add W&B import
import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import shared_memory

#os.environ["WANDB_MODE"] = "offline"

def get_sim_number(filepath):
    filename = os.path.basename(filepath)
    number_part = filename.split('_')[-1]
    number = int(number_part.split('.')[0])
    return number

class PairedDataset(Dataset):
    def __init__(self, x_dir, y_dir, start_idx, end_idx):
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        x_files, y_files = [], []
        for ji in range(start_idx, end_idx):
            x_files.append(os.path.join(y_dir,  f'{ji}/halo_sentence_LH_{ji}.npy'))
            y_files.append(os.path.join(x_dir, f'{ji}/dmo_fields_subvols_grid_8_LH_{ji}.npy'))
            

        assert len(x_files) == len(y_files), "X and Y have different number of files"

        total_files = len(x_files)
        files_per_rank = total_files // world_size  # ceil
        start_idx = rank * files_per_rank
        end_idx = start_idx + files_per_rank

        print(f"Rank {rank}: Loading files from index {start_idx} to {end_idx} out of {total_files}")

        x_files = x_files[start_idx:end_idx]
        y_files = y_files[start_idx:end_idx]

        tempx = np.load(x_files[0])
        tempy = np.load(y_files[0])
        tempy = np.moveaxis(tempy, -1, 1)
        nx = tempx.shape[0]
        ny = tempy.shape[0]
        x_seq = tempx.shape[1]
        y_rest_shape = tempy.shape[1:]

        total_x = nx * len(x_files)
        total_y = ny * len(y_files)

        self.xdata = torch.empty((total_x, x_seq), dtype=torch.long)
        self.ydata = torch.empty((total_y, *y_rest_shape), dtype=torch.float)

        x_start = 0
        y_start = 0
        for xf, yf in zip(x_files, y_files):
            x_arr = torch.from_numpy(np.load(xf)).long()
            y_arr = torch.from_numpy(np.load(yf)).float()
            y_arr = torch.moveaxis(y_arr, -1, 1)  # move last dimension to second position
            self.xdata[x_start:x_start + nx].copy_(x_arr)
            self.ydata[y_start:y_start + ny].copy_(y_arr)
            x_start += nx
            y_start += ny

        del x_arr, y_arr
        gc.collect()

    def __len__(self):
        return self.xdata.shape[0]

    def __getitem__(self, idx):
        return self.xdata[idx], self.ydata[idx]


def parse_args():
    parser = argparse.ArgumentParser(description='Training script with key-value arguments')
    
    # Define arguments with their default values and types
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=10,
                        help='Maximum iterations')
    parser.add_argument('--n_embd', type=int, default=384,
                        help='Embedding dimension')                                                                     
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        help='Maximum iterations')    
    parser.add_argument('--cnn_type', type=str, default='vit_cbam',
                        help='Maximum iterations')  
    parser.add_argument('--gauss_delta', type=float, default=0.1,
                        help='Gaussian delta for loss smoothing')                          
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    add_space_token = False
    subsel_type = 'all'
    learning_rate = args.learning_rate
    max_iters = args.max_iters
    loss_type = args.loss_type
    cnn_type = args.cnn_type
    n_embd = args.n_embd
    gauss_delta = args.gauss_delta
    print(f"add_space_token = {add_space_token}, subsel_type = {subsel_type}, learning_rate = {learning_rate}, max_iters = {max_iters}, loss_type = {loss_type}, cnn_type = {cnn_type}")
    


def train():
    
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    device_type = 'cuda'
    dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    print(int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", rank=int(os.environ.get("RANK", 0)), world_size=int(os.environ.get("WORLD_SIZE", 1)), device_id=torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'))
    rank = dist.get_rank()
    Ndevices = torch.cuda.device_count()
    torch.cuda.empty_cache()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"Start running basic DDP example on rank {rank}.")

    meta_f = pk.load(open('/mnt/ceph/users/spandey/discodj_runs/halos_story_nsel_32768/sentence_params.pkl','rb'))
    start_token = meta_f['start_token']
    pad_token = int(meta_f['pad_token'])
    end_token = meta_f['end_token']
    nvocab_total = end_token + 1
    max_sentence_length = meta_f['max_sentence_length']

    n_head = 12
    n_layer = 8
    n_layers_vit = 4
    n_heads_vit = 8
    dropout = 0.
    vocab_size = nvocab_total
    block_size = max_sentence_length - 1 + 1
    
    if dist.get_rank() == 0:
        print(f"block_size = {block_size}, vocab_size = {vocab_size}, pad_token = {pad_token}, max_sentence_length = {max_sentence_length}")
        print(f"nembd = {n_embd}, nhead = {n_head}, nlayer = {n_layer}, dropout = {dropout}")

    dmo_cond_embed_type = 'vit'
    layers_types =  ['res_cbam']
    #layers_types =  ['cnn']
    # patch_size = 4 # subgrid = 8
    patch_size = 4

    # Om_min = 0.1
    # Om_max = 0.5
    # sigma8_min = 0.6
    # sigma8_max = 1.0
    # nbin = 64
    
    HaloConfig = {'block_size': block_size, 'vocab_size': vocab_size, 'n_layer': n_layer, 
                    'n_head': n_head, 'n_embd': n_embd, 'dropout': dropout, 
                    'bias': False, 'ksize': 3, 'density_grid_in': 8, 'density_grid_out': 4, 
                    'ninp_density': 18, 'pad_token': pad_token, 'flash': True,
                    'dmo_cond_embed_type':dmo_cond_embed_type, 'layers_types':layers_types,
                    'patch_size':patch_size,'loss_type':loss_type,'gauss_delta':gauss_delta,'device':f'cuda:{local_rank}',
                    'n_layers_vit': n_layers_vit, 'n_heads_vit': n_heads_vit,
                    'max_nhalo':36, 'nprops': 8, 'end_token': end_token, 'start_token': start_token}

    
    model = HaloDecoderModel(HaloConfig).to(local_rank)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    '''
    
    cp_name = '/work/hdd/bdne/yzhang116/checkpoints_36/checkpoint_36_hnumv3_cross_entropy_epoch_9_batch_4000_embed_1024_batch_800_lrmax_2e-05_lrmin_1e-06_layer_16_head_16_layervit_4_headvit_8_patch_4.pt'
    checkpoint = torch.load(cp_name, map_location=f'cuda:{local_rank}')
    HaloConfig = checkpoint['config']
    model = HaloDecoderModel(HaloConfig).to(local_rank)    
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16')) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    '''


    model = DDP(model, device_ids=[local_rank]) #, find_unused_parameters=True) # DistributedDataParallel

    model.train()
    if rank == 0: print(f"Init model and loaded to GPU", flush=True)    
    train_num_start = 0
    train_num_end = 10
    val_num_start = 10
    val_num_end = 20
    shuffle = False

    #dmo_dir = '/work/hdd/bdne/yzhang116/dmo_fields_gridsbox16/train/'
    dmo_dir = '/mnt/ceph/users/spandey/discodj_runs/rhog_LH_np_512_nsnap_3_nsel_32768/'
    halo_sentence_dir = '/mnt/ceph/users/spandey/discodj_runs/halos_story_nsel_32768/'
    #dmo_dir_val = '/work/hdd/bdne/yzhang116/dmo_fields_gridsbox16/validation/'
    # dmo_dir_val = '/work/hdd/bdne/yzhang116/dmo_fields_4096_snap3/validation/'
    # halo_sentence_dir_val = '/work/hdd/bdne/yzhang116/halo_sentences_36/validation/'
    
    
    
    vali_dataset = PairedDataset(halo_sentence_dir, dmo_dir,val_num_start, val_num_end)
    batch_size_val = 125
    batch_num_val = len(vali_dataset)//(batch_size_val)
    vali_dataloader = DataLoader(vali_dataset, batch_size=batch_size_val, shuffle=shuffle, num_workers=1, pin_memory=True, drop_last=True)
    print("rank %d, len(validata) = %d"%(dist.get_rank(), len(vali_dataset)), flush=True)

    train_dataset = PairedDataset(halo_sentence_dir, dmo_dir,train_num_start, train_num_end)
    batch_size = 125
    batch_num = len(train_dataset)//(batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True, drop_last=True)
    print("rank %d, len(traindata) = %d"%(dist.get_rank(), len(train_dataset)), flush=True)
    


    lr_min = 1e-5
    lr_max = 1e-4 #learning_rate
    warmup_steps =2000

    def lr_lambda(step, total_steps):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps) * lr_max
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (step-warmup_steps) / total_steps))

    if dist.get_rank() == 0:
        wandb.login(key="ef0e0f2bb165256e3c69f0546adfd59b50b05f2c")         
        wandb.init(
            project='quijote_v2',
            name='hnumv3_%s_embed%d_lrmax%e_batch%d'%(loss_type,n_embd,lr_max,batch_size*Ndevices),
            config={
                "epochs": max_iters,
                "batch_size": batch_size*Ndevices,
                "loss_type": loss_type,
                'gauss_delta': gauss_delta,
                "lr_max": lr_max,
                "lr_min": lr_min,
                "training_cosmology": cosnum,
                "n_embd": n_embd,
                "dropout": dropout,
                "cnn_type": cnn_type,
                "n_head": n_head,
                "n_layer": n_layer,
                "patch_size": patch_size,
                'layers_types': layers_types,
                "n_layers_vit": n_layers_vit,
                "n_heads_vit": n_heads_vit,
                "cosnum": cosnum,
                
            }
        )
    # Log model architecture
    if dist.get_rank() == 0:
        wandb.watch(model, log="all", log_freq=500)
    # nepochs = max_iters 
        #train_log = open("/work/hdd/bdne/yzhang116/logs/train_36_%s_embed%d_batch%d_lrmin%e_lrmax%e.log"%(loss_type,n_embd,batch_size,lr_min,lr_max), "w")    
        #val_log = open("/work/hdd/bdne/yzhang116/logs/validation_36_%s_embed%d_batch%d_lrmin%e_lrmax%e.log"%(loss_type,n_embd,batch_size,lr_min,lr_max), "w")
    best_loss = 1e20
    for jepoch in range(0, max_iters):
        for batch_idx, batch_data in enumerate(train_dataloader):
            DM = batch_data[1].to(device, non_blocking=True)
            # DM = torch.moveaxis(DM, -1, 1)  # move the last dimension to the second position
            #print("DM shape:", DM.shape,flush=True)  # should be (B, C, D, H, W)
            gal_tokens = batch_data[0].to(device, non_blocking=True)
            B = gal_tokens.shape[0]
            #print("gal_tokens shape:", gal_tokens.shape,flush=True)  # should be (B, L)
            #X = gal_tokens[:, :-1].to(torch.long)
            #Y = gal_tokens[:, 1:].to(torch.long)
            #Y[:, :2] = pad_token  # cosmology tokens do not contribute to loss
            end_token_index = (gal_tokens == end_token).nonzero(as_tuple=True)[1] #(b,)
            n_halo_actual = (end_token_index - 5) // 8  # shape: (b,)
            n_halo_actual = torch.clamp(n_halo_actual, min=0, max=64)
            X = torch.cat([gal_tokens[:,:5],n_halo_actual.unsqueeze(1),gal_tokens[:,5:-1]],dim=1).long()
            Y = torch.cat([torch.zeros(B,1,device=device), gal_tokens[:,1:]],dim=1).long()
            Y[:, :5] = pad_token
            Y[:, 5] = n_halo_actual

            mask = torch.logical_not(X != pad_token)
            masked_logits = torch.zeros(mask.shape, device=X.device, dtype=torch.float32)
            MASK = masked_logits.masked_fill(mask, float('-inf'))[:,None,:]

            #lr = cosine_lr(batch_idx, batch_num-1, lr_min, learning_rate/((1 + jepoch)**0.75))
            lr = lr_lambda(jepoch*batch_num+batch_idx, max_iters*batch_num)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            with ctx:
                loss, _ = model(X, DM, 
                                maskd=MASK, 
                                targets=Y)
                scaler.scale(loss).backward()
            loss_tensor = torch.tensor(loss.item(), device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_mean_here = loss_tensor.item()
            #dist.all_reduce(loss_array, op=dist.ReduceOp.AVG)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if dist.get_rank() == 0:
                wandb.log({
                    "train/batch_total_loss": loss_mean_here,
                    "train/epoch": jepoch,
                    "train/lr": lr,
                    }, step=jepoch*batch_num+batch_idx)


                print("epoch: %d step %d Mean loss of all gpus for this batch: "%(jepoch,batch_idx), loss_mean_here)
                print("lr: ", lr)
                #train_log.write(f"{jepoch*batch_num+batch_idx}\t{loss_mean_here:.8f}\n")
                #train_log.flush()
        
            if batch_idx % 1000 == 0: # and batch_idx > 0:
                model.eval()
                loss_mean_here = 0
                
                with torch.no_grad():
                    for val_batch_idx, val_batch_data in enumerate(vali_dataloader):
                        DM_val = val_batch_data[1].to(device, non_blocking=True)
                        # DM_val = torch.moveaxis(DM_val, -1, 1)  # move the last dimension to the second position
                        gal_tokens_val = val_batch_data[0].to(device, non_blocking=True)
                        B = gal_tokens_val.shape[0]
                        #X_val = gal_tokens_val[:, :-1].to(torch.long)
                        #Y_val = gal_tokens_val[:, 1:].to(torch.long)
                        #Y_val[:, :2] = pad_token
                        end_token_index_val = (gal_tokens_val == end_token).nonzero(as_tuple=True)[1] #(b,)
                        n_halo_actual_val = (end_token_index_val - 5) // 8  # shape: (b,)
                        n_halo_actual_val = torch.clamp(n_halo_actual_val, min=0, max=64)
                        X_val = torch.cat([gal_tokens_val[:,:5],n_halo_actual_val.unsqueeze(1),gal_tokens_val[:,5:-1]],dim=1).long()
                        Y_val = torch.cat([torch.zeros(B,1,device=device), gal_tokens_val[:,1:]],dim=1).long()
                        Y_val[:, :5] = pad_token
                        Y_val[:, 5] = n_halo_actual_val
                        #cut = get_curr_seq_len(jepoch*batch_num+batch_idx)
                        #X_val = X_val[:,:cut].contiguous()
                        #Y_val = Y_val[:,:cut].contiguous()
                        mask_val = torch.logical_not(X_val != pad_token)
                        masked_logits_val = torch.zeros(mask_val.shape, device=X_val.device, dtype=torch.float32)
                        MASK_val = masked_logits_val.masked_fill(mask_val, float('-inf'))[:,None,:]
                        
                        with ctx:
                            val_loss,_ = model(X_val, DM_val, 
                                                maskd=MASK_val, 
                                                targets=Y_val)
                        loss_tensor = torch.tensor(val_loss.item(), device=device)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                        loss_mean_here += loss_tensor.item()

                    loss_mean_here /= (val_batch_idx + 1)
                if dist.get_rank() == 0:
                    wandb.log({
                        "validation/total_loss": loss_mean_here,
                        "validation/epoch": jepoch,
                        }, step=jepoch*batch_num+batch_idx)
                    print("Validation epoch: %d batch_idx: %d Mean loss for this epoch: "%(jepoch, batch_idx), loss_mean_here)
                    #val_log.write(f"{jepoch*batch_num+batch_idx}\t{loss_mean_here:.8f}\n")
                    #val_log.flush()
                model.train()
                    # if loss is less than previous best, save the checkpoint:
                if loss_mean_here < best_loss:
                    best_loss = loss_mean_here
                    if dist.get_rank() == 0:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'global_step': jepoch,
                            'loss': best_loss,
                            'config': HaloConfig,
                        }
                        check_point_name = f'/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/checkpoints/checkpoint_hnumv3_{loss_type}_epoch_{jepoch}_batch_{batch_idx}_embed_{n_embd}_batch_{batch_size*Ndevices}_lrmax_{lr_max}_lrmin_{lr_min}_layer_{n_layer}_head_{n_head}_layervit_{n_layers_vit}_headvit_{n_heads_vit}_patch_{patch_size}_dropout_{dropout}_shuffle_{shuffle}.pt'
                        torch.save(checkpoint, check_point_name)
                        print(f"Checkpoint saved at epoch {jepoch} with loss {best_loss:.4f}: {check_point_name}", flush=True)
        
    if dist.get_rank() == 0:
        wandb.finish()
        print("Training finished.")
        #train_log.close()
        #val_log.close()
    
    train_dataset.cleanup()
    vali_dataset.cleanup()
    dist.destroy_process_group()

if __name__ == "__main__":
    train()