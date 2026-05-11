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
from model_enc_dec_cos import *
from gen_func import *
from performance import *
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
import time

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

#os.environ["WANDB_MODE"] = "offline"

def get_sim_number(filepath):
    filename = os.path.basename(filepath)
    number_part = filename.split('_')[-1]
    number = int(number_part.split('.')[0])
    return number


class ExternalInputIterator:
    def __init__(self, input_dir, label_dir, batch_size, shard_id, num_shards, shuffle=False):
        files_in_inp_dir = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]
        self.dm_fields_files = sorted(files_in_inp_dir, key=get_sim_number)
        files_in_label_dir = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')]
        self.gal_prop_files = sorted(files_in_label_dir, key=get_sim_number)
        total = len(self.dm_fields_files)
        self.shuffle = shuffle
        # print(total, len(self.gal_prop_files))
        assert total == len(self.gal_prop_files)
        
        # Shard the data: split in round-robin (recommended by DALI)
        self.indices = np.arange(total)[shard_id::num_shards]   # total is the number of LH sims
        if shuffle:
            np.random.shuffle(self.indices)
        self.data_set_len = len(self.indices)
        self.batch_size = batch_size
    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    def __next__(self):
        if self.i >= self.data_set_len:
            self.i = 0
            raise StopIteration
        batch_inputs, batch_labels = [], []
        for _ in range(self.batch_size):  # each GPU gets batch_size samples
            if self.i >= self.data_set_len:
                break
            idx = self.indices[self.i]
            batch_inputs.append(np.load(self.dm_fields_files[idx])) # dm_fields_files[idx]: (nrand_sel_box,grid_sbox,grid_sbox,grid_sbox,6*snap)
            batch_labels.append(np.load(self.gal_prop_files[idx])) # (nrand_sel_box, max_sentence_length)
            # print(idx)
            nrep = batch_inputs[-1].shape[0]
            self.i += 1
        return (batch_inputs, batch_labels)
    @property
    def size(self):
        return self.data_set_len

def create_dali_pipeline(input_dir, label_dir, batch_size, num_threads, device_id, shard_id, num_shards, shuffle=False):
    eii = ExternalInputIterator(input_dir, label_dir, batch_size, shard_id, num_shards, shuffle=shuffle)
    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    def npy_pipeline():
        dmo_fields, gal_tokens = fn.external_source(source=eii, num_outputs=2)
        return dmo_fields.gpu(), gal_tokens.gpu()
    pipe = npy_pipeline()
    pipe.build()
    return pipe, eii.size


def parse_args():
    parser = argparse.ArgumentParser(description='Training script with key-value arguments')
    
    # Define arguments with their default values and types
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=5,
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
    Ndevices = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.empty_cache()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    job_id = int(os.environ.get("SLURM_JOB_ID"))

    print(f"Start running basic DDP example on rank {rank}.")

    meta_f = pk.load(open('/work/nvme/bdne/yzhang116/quijote_halos/sentence_params.pkl','rb'))
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

    dmo_dir = '/work/nvme/bdne/yzhang116/quijote_fields/train/'
    halo_sentence_dir = '/work/nvme/bdne/yzhang116/quijote_halos/train/'
    dmo_dir_val = '/work/nvme/bdne/yzhang116/quijote_fields/validation/'
    halo_sentence_dir_val = '/work/nvme/bdne/yzhang116/quijote_halos/validation/'

    batch_size = 2
    pipeline, n_samples_per_rank = create_dali_pipeline(
        dmo_dir, halo_sentence_dir, batch_size, num_threads=4, device_id=local_rank,
        shard_id=dist.get_rank(), num_shards=int(os.environ.get("WORLD_SIZE", 1)), shuffle=True)
    dali_iterator = DALIGenericIterator(
        [pipeline],
        ['DM_fields', 'gal_tokens'],
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True
    )

    batch_size_val = 2
    pipeline_val, n_samples_per_rank_val = create_dali_pipeline(
        dmo_dir_val, halo_sentence_dir_val, batch_size_val, num_threads=4, device_id=local_rank,
        shard_id=dist.get_rank(), num_shards=int(os.environ.get("WORLD_SIZE", 1)), shuffle=True)
    dali_iterator_val = DALIGenericIterator(
        [pipeline_val],
        ['DM_fields', 'gal_tokens'],
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True
    )

    sub_batch_size = 200
    sub_batch_size_val = 200
    subbox_per_file = 20000
    batch_num = n_samples_per_rank // batch_size
    nsub_batches = subbox_per_file * batch_size // sub_batch_size
    
    

    lr_min = 1e-5
    lr_max = 1e-4 #learning_rate
    warmup_steps = 2000

    def lr_lambda(step, total_steps):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps) * lr_max
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (step-warmup_steps) / (total_steps - warmup_steps)))

    if dist.get_rank() == 0:
        wandb.login(key="22b8042f587b46afa3f77fa124d0a1b135bd5dd1")         
        wandb.init(
            project='quijote',
            name='lm_%s_embed%d_lrmax%e_batch%d'%(loss_type,n_embd,lr_max,sub_batch_size*Ndevices),
            config={
                "epochs": max_iters,
                "batch_size": sub_batch_size*Ndevices,
                "loss_type": loss_type,
                'gauss_delta': gauss_delta,
                "lr_max": lr_max,
                "lr_min": lr_min,
                "n_embd": n_embd,
                "dropout": dropout,
                "cnn_type": cnn_type,
                "n_head": n_head,
                "n_layer": n_layer,
                "patch_size": patch_size,
                'layers_types': layers_types,
                "n_layers_vit": n_layers_vit,
                "n_heads_vit": n_heads_vit,
            }
        )
    # Log model architecture
    if dist.get_rank() == 0:
        wandb.watch(model, log="all", log_freq=500)
        world_size = dist.get_world_size()
        log_ps = np.zeros((world_size,2,443))
        log_hmf = np.zeros((world_size,2,19))
    # nepochs = max_iters 
        #train_log = open("/work/hdd/bdne/yzhang116/logs/train_36_%s_embed%d_batch%d_lrmin%e_lrmax%e.log"%(loss_type,n_embd,batch_size,lr_min,lr_max), "w")    
        #val_log = open("/work/hdd/bdne/yzhang116/logs/validation_36_%s_embed%d_batch%d_lrmin%e_lrmax%e.log"%(loss_type,n_embd,batch_size,lr_min,lr_max), "w")
    best_loss = 1e20
    sub_step = 0
    for jepoch in range(0, max_iters):
        for batch_idx, batch_data in enumerate(dali_iterator):
            DM = batch_data[0]['DM_fields']
            DM = torch.flatten(DM, start_dim=0, end_dim=1)
            DM = torch.moveaxis(DM, -1, 1)  # move the last dimension to the second position
            #print("DM shape:", DM.shape,flush=True)  # should be (B, C, D, H, W)
            gal_tokens = batch_data[0]['gal_tokens']
            gal_tokens = torch.flatten(gal_tokens, start_dim=0, end_dim=1)
            B = gal_tokens.shape[0]
            #print("gal_tokens shape:", gal_tokens.shape,flush=True)  # should be (B, L)
            X = gal_tokens[:, :-1].to(torch.long)
            Y = gal_tokens[:, 1:].to(torch.long)
            Y[:, :5] = pad_token  # cosmology tokens do not contribute to loss

            mask = torch.logical_not(X != pad_token)
            masked_logits = torch.zeros(mask.shape, device=X.device, dtype=torch.float32)
            MASK = masked_logits.masked_fill(mask, float('-inf'))[:,None,:]

            all_inds = np.arange(B)
            np.random.shuffle(all_inds)
            for js in range(nsub_batches):
                ind_js = all_inds[js*sub_batch_size:(js+1)*sub_batch_size]
                #lr = cosine_lr(batch_idx, batch_num-1, lr_min, learning_rate/((1 + jepoch)**0.75))
                lr = lr_lambda(sub_step, max_iters*batch_num*nsub_batches)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                with ctx:
                    loss = model(X[ind_js], DM[ind_js], 
                                    maskd=MASK[ind_js], 
                                    targets=Y[ind_js])
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
                        }, step=sub_step)

                    print("epoch: %d step %d Mean loss of all gpus for this batch: "%(jepoch,sub_step), loss_mean_here)
                    print("lr: ", lr)
                #train_log.write(f"{jepoch*batch_num+batch_idx}\t{loss_mean_here:.8f}\n")
                #train_log.flush()
                sub_step += 1

            if sub_step % (nsub_batches*5) == 0:
                model.eval()
                loss_mean_here = 0
                count_val = 0
                with torch.no_grad():
                    for batch_idx_val, batch_data_val in enumerate(dali_iterator_val):
                        DM_val = batch_data_val[0]['DM_fields']
                        DM_val = torch.flatten(DM_val, start_dim=0, end_dim=1)
                        DM_val = torch.moveaxis(DM_val, -1, 1)  # move the last dimension to the second position
                        gal_tokens_val = batch_data_val[0]['gal_tokens']
                        gal_tokens_val = torch.flatten(gal_tokens_val, start_dim=0, end_dim=1)
                        B = gal_tokens_val.shape[0]
                        X_val = gal_tokens_val[:, :-1].to(torch.long)
                        Y_val = gal_tokens_val[:, 1:].to(torch.long)
                        Y_val[:, :5] = pad_token
                        mask_val = torch.logical_not(X_val != pad_token)
                        masked_logits_val = torch.zeros(mask_val.shape, device=X_val.device, dtype=torch.float32)
                        MASK_val = masked_logits_val.masked_fill(mask_val, float('-inf'))[:,None,:]
                        all_inds = np.arange(B)
                        nsub_batches_val = B // sub_batch_size_val
                        for js in range(nsub_batches_val):
                            ind_js = all_inds[js*sub_batch_size_val:(js+1)*sub_batch_size_val]
                            with ctx:
                                val_loss = model(X_val[ind_js], DM_val[ind_js], 
                                                    maskd=MASK_val[ind_js], 
                                                    targets=Y_val[ind_js])
                            loss_tensor = torch.tensor(val_loss.item(), device=device)
                            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                            loss_mean_here += loss_tensor.item()
                            count_val += 1

                    loss_mean_here /= count_val
                if dist.get_rank() == 0:
                    wandb.log({
                        "validation/total_loss": loss_mean_here,
                        "validation/epoch": jepoch,
                        }, step=sub_step)
                    print("Validation epoch: %d step: %d Mean loss for this epoch: "%(jepoch, sub_step), loss_mean_here)
                    #val_log.write(f"{jepoch*batch_num+batch_idx}\t{loss_mean_here:.8f}\n")
                    #val_log.flush()
                    # if loss is less than previous best, save the checkpoint:
                if loss_mean_here < best_loss:
                    best_loss = loss_mean_here
                if loss_mean_here > best_loss + 1:
                    print("Validation loss increased significantly, exiting training to prevent overfitting.", flush=True)
                    exit(0)
                if loss_mean_here < best_loss + 0.1:
                    if dist.get_rank() == 0:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'global_step': jepoch,
                            'loss': best_loss,
                            'config': HaloConfig,
                        }
                        base_dir = "/work/hdd/bdne/yzhang116/checkpoints_quijote"
                        job_dir = os.path.join(base_dir, f"{job_id}")
                        os.makedirs(job_dir, exist_ok=True)
                        check_point_name = os.path.join(job_dir, f'checkpoint_lm_{loss_type}_epoch_{jepoch}_step_{sub_step}_embed_{n_embd}_batch_{sub_batch_size*Ndevices}_lrmax_{lr_max}_lrmin_{lr_min}_layer_{n_layer}_head_{n_head}_layervit_{n_layers_vit}_headvit_{n_heads_vit}_patch_{patch_size}_dropout_{dropout}.pt')
                        torch.save(checkpoint, check_point_name)
                        print(f"Checkpoint saved at epoch {jepoch} with loss {best_loss:.4f}: {check_point_name}", flush=True)

                simids = [1414,660,66,975,1565,1625,1974,1043]
                num_true = [140330,746524,566607,340981,691685,218454,460621,1003888]
                world_size = dist.get_world_size()
                simid = simids[int(dist.get_rank())]
                start_time = time.time()
                gen_data = gen(simid, model)
                end_time = time.time()
                print(f"Generation time for simid {simid} on rank {dist.get_rank()}: {(end_time-start_time)/60} mins", flush=True)
                hmf,ps,num_infer= get_ratios(simid,gen_data) 
                obj = {"rank": dist.get_rank(), "hmf": hmf, "pk": ps, "num_infer": num_infer}
                dist.barrier()
                if dist.get_rank() == 0:
                    gathered = [None for _ in range(world_size)]
                else:
                    gathered = None
                dist.gather_object(obj, gathered, dst=0)

                if dist.get_rank() == 0:
                    for obj in gathered:
                        r = obj["rank"]
                        hmf = obj["hmf"]
                        ps = obj["pk"]
                        num_infer = obj["num_infer"]
                        log_ps[r,1] = ps[1]
                        log_hmf[r,1] = hmf[1]
                        log_ps[r,0] = ps[0]
                        log_hmf[r,0] = hmf[0]
                        print(f"Number of inferred halos: {num_infer}", flush=True)

                        wandb.log(
                            {   '''
                                f"Statistics/PowerSpectra_GPU{r}": wandb.plot.line_series(
                                xs=np.log10(ps[0]),
                                ys=log_ps[r].tolist(),
                                keys=[f"step_{sub_step - i*5000}" for i in range(5)][::-1],
                                title=f"Power Spectrum {simids[r]}",
                                xname="log10(k) [h/Mpc]",
                                ),
                                f"Statistics/HMF_GPU{r}": wandb.plot.line_series(
                                xs=hmf[0],
                                ys=log_hmf[r].tolist(),
                                keys=[f"step_{sub_step - i*5000}" for i in range(5)][::-1],
                                title=f"HMF {simids[r]}",
                                xname="log10(M) [Msun/h]",
                                ),
                                '''
                                f"Statistics/NumHalos_GPU{r}_true{num_true[r]}": num_infer,
                            },
                            step=sub_step,
                        )
                    base_dir = "/u/yzhang116/NN/plots"
                    plot_dir = os.path.join(base_dir, f"{job_id}")
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_name = os.path.join(plot_dir, f"performance_step_{sub_step}_{job_id}_lm.png")
                    plot_log(log_ps,log_hmf,num_true,plot_name)
                model.train()
                        
        
    if dist.get_rank() == 0:
        wandb.finish()
        print("Training finished.")
        #train_log.close()
        #val_log.close()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    train()