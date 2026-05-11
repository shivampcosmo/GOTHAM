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
from model_enc_dec_cos_fast_xfirst import *
from gen_func_xfirst import *
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
import wandb
import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import shared_memory
import time

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def get_sim_number(filepath):
    filename = os.path.basename(filepath)
    number_part = filename.split('_')[-1]
    number = int(number_part.split('.')[0])
    return number


def reorder_halo_tokens(gal_tokens, nprops=8, max_nhalo=36):
    """
    Reorder gal_tokens from halo-grouped to property-grouped layout.

    Old: [START, cosmo×5, N_halos,  x1,y1,z1,...,p8_1,  x2,...,p8_2,  ..., END, PAD...]
    New: [START, cosmo×5, N_halos,  x1,...,xN,  y1,...,yN, ..., p8_1,...,p8_N,  END, PAD...]

    For each sample, extract the N*nprops real halo tokens, reshape (N, nprops),
    transpose to (nprops, N), flatten, and write back. END stays at same position.
    """
    B, L = gal_tokens.shape
    n_halo_actual = gal_tokens[:,6]  # (B,)
    halo_token_max = max_nhalo * nprops
    n_halo_actual = n_halo_actual.unsqueeze(1).repeat(1, halo_token_max)  # (B, max_nhalo*nprops)
    j = torch.arange(halo_token_max, device=gal_tokens.device).unsqueeze(0)  # (1, max_nhalo*nprops)
    safe_n = n_halo_actual.clamp(min=1)
    halo_id = j % safe_n
    prop_id = j // safe_n
    valid = (halo_id < n_halo_actual) & (prop_id < nprops)  # (B, max_nhalo*nprops)
    original_indices = halo_id * nprops + prop_id + 7  # (B, max_nhalo*nprops)
    original_indices = torch.where(valid, original_indices, torch.zeros_like(original_indices))  # (B, max_nhalo*nprops)
    batch_indices = torch.arange(B, device=gal_tokens.device).unsqueeze(1).repeat(1, halo_token_max)  # (B, max_nhalo*nprops)
    new_tokens = gal_tokens[batch_indices, original_indices]  # (B, max_nhalo*nprops)
    gal_tokens[:,7:7+halo_token_max] = torch.where(valid, new_tokens, gal_tokens[:,7:7+halo_token_max])  # write back to gal_tokens

    return gal_tokens


class ExternalInputIterator:
    def __init__(self, input_dir, label_dir, batch_size, shard_id, num_shards, shuffle=False):
        files_in_inp_dir = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]
        self.dm_fields_files = sorted(files_in_inp_dir, key=get_sim_number)
        files_in_label_dir = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')]
        self.gal_prop_files = sorted(files_in_label_dir, key=get_sim_number)
        total = len(self.dm_fields_files)
        self.shuffle = shuffle
        assert total == len(self.gal_prop_files)
        self.indices = np.arange(total)[shard_id::num_shards]
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
        for _ in range(self.batch_size):
            if self.i >= self.data_set_len:
                break
            idx = self.indices[self.i]
            batch_inputs.append(np.load(self.dm_fields_files[idx]))
            batch_labels.append(np.load(self.gal_prop_files[idx]))
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
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--cnn_type', type=str, default='vit_cbam')
    parser.add_argument('--gauss_delta', type=float, default=0.1)
    parser.add_argument('--cp_id', type=int, default=1872236)
    parser.add_argument('--cp_step', type=int, default=7050)
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
    cp_id = args.cp_id
    cp_step = args.cp_step
    print(f"add_space_token = {add_space_token}, subsel_type = {subsel_type}, "
          f"learning_rate = {learning_rate}, max_iters = {max_iters}, "
          f"loss_type = {loss_type}, cnn_type = {cnn_type}")
    print("cp_id = ", cp_id, " cp_step = ", cp_step)


def train():

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda'
    dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    print(int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)),
          int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(
        "nccl",
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        device_id=torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}')
    )
    rank = dist.get_rank()
    Ndevices = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.empty_cache()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    job_id = int(os.environ.get("SLURM_JOB_ID"))

    print(f"Start running basic DDP example on rank {rank}.")

    meta_f = pk.load(open('/work/nvme/bdne/yzhang116/quijote_halos/sentence_params.pkl', 'rb'))
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
        print(f"block_size = {block_size}, vocab_size = {vocab_size}, "
              f"pad_token = {pad_token}, max_sentence_length = {max_sentence_length}")
        print(f"nembd = {n_embd}, nhead = {n_head}, nlayer = {n_layer}, dropout = {dropout}")
        print("model: model_enc_dec_cos_prop_grouped")

    dmo_cond_embed_type = 'vit'
    layers_types = ['res_cbam']
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
    cp_name = (f'/work/hdd/bdne/yzhang116/checkpoints_quijote/{cp_id}/'
               f'checkpoint_cos_cross_entropy_epoch_0_step_{cp_step}_embed_384_batch_800'
               f'_lrmax_1e-05_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt')
    print("Loading checkpoint: ", cp_name, flush=True)
    checkpoint = torch.load(cp_name, map_location=f'cuda:{local_rank}')
    HaloConfig = checkpoint['config']
    model = HaloDecoderModel(HaloConfig).to(local_rank)
    state_dict = checkpoint["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    '''

    model = DDP(model, device_ids=[local_rank])
    model.train()
    if rank == 0:
        print(f"Init model and loaded to GPU", flush=True)

    dmo_dir = '/work/hdd/bdne/yzhang116/quijote_full/quijote_fields/'
    halo_sentence_dir = '/work/hdd/bdne/yzhang116/quijote_full/quijote_halos/'

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
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (step-warmup_steps) / total_steps))

    if dist.get_rank() == 0:
        wandb.login(key="22b8042f587b46afa3f77fa124d0a1b135bd5dd1")
        wandb.init(
            project='quijote',
            name='xfirst_lrmax%e_batch%d' % (lr_max, sub_batch_size * Ndevices),
            config={
                "epochs": max_iters,
                "batch_size": sub_batch_size * Ndevices,
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
                "job_id": job_id,
                "model": "model_enc_dec_cos_xfirst",
            }
        )

    if dist.get_rank() == 0:
        wandb.watch(model, log="all", log_freq=500)
        world_size = dist.get_world_size()
        log_ps = np.zeros((world_size, 2, 443))
        log_hmf = np.zeros((world_size, 2, 19))

    best_loss = 1e20
    sub_step = 0
    nprops = HaloConfig['nprops']
    max_nhalo = HaloConfig['max_nhalo']

    for jepoch in range(0, max_iters):
        for batch_idx, batch_data in enumerate(dali_iterator):
            DM = batch_data[0]['DM_fields']
            DM = torch.flatten(DM, start_dim=0, end_dim=1)
            DM = torch.moveaxis(DM, -1, 1)

            gal_tokens = batch_data[0]['gal_tokens']
            gal_tokens = torch.flatten(gal_tokens, start_dim=0, end_dim=1)

            # ---- Reorder halo tokens from halo-grouped to property-grouped layout ----
            gal_tokens = reorder_halo_tokens(gal_tokens)
            print("Reordered gal_tokens to property-grouped layout", flush=True)
            # -------------------------------------------------------------------------

            B = gal_tokens.shape[0]
            X = gal_tokens[:, :-1].to(torch.long)
            Y = gal_tokens[:, 1:].to(torch.long)
            Y[:, :5] = pad_token  # cosmology tokens do not contribute to loss

            mask = torch.logical_not(X != pad_token)
            masked_logits = torch.zeros(mask.shape, device=X.device, dtype=torch.float32)
            MASK = masked_logits.masked_fill(mask, float('-inf'))[:, None, :]

            all_inds = np.arange(B)
            np.random.shuffle(all_inds)
            for js in range(nsub_batches):
                ind_js = all_inds[js * sub_batch_size:(js + 1) * sub_batch_size]
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if dist.get_rank() == 0:
                    wandb.log({
                        "train/batch_total_loss": loss_mean_here,
                    }, step=sub_step)
                    print("epoch: %d step %d Mean loss of all gpus for this batch: " % (jepoch, sub_step),
                          loss_mean_here)
                sub_step += 1

                if sub_step % 500 == 0 and sub_step > 0:
                    model.eval()
                    if dist.get_rank() == 0:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'global_step': jepoch,
                            'loss': loss_mean_here,
                            'config': HaloConfig,
                        }
                        base_dir = "/work/hdd/bdne/yzhang116/checkpoints_quijote"
                        job_dir = os.path.join(base_dir, f"{job_id}")
                        os.makedirs(job_dir, exist_ok=True)
                        check_point_name = os.path.join(
                            job_dir,
                            f'checkpoint_cos_{loss_type}_epoch_{jepoch}_step_{sub_step}'
                            f'_embed_{n_embd}_batch_{sub_batch_size * Ndevices}'
                            f'_lrmax_{lr_max}_lrmin_{lr_min}_layer_{n_layer}_head_{n_head}'
                            f'_layervit_{n_layers_vit}_headvit_{n_heads_vit}'
                            f'_patch_{patch_size}_dropout_{dropout}.pt'
                        )
                        torch.save(checkpoint, check_point_name)
                        print(f"Checkpoint saved at epoch {jepoch} with loss {loss_mean_here:.4f}: "
                              f"{check_point_name}", flush=True)

                    simids = [1414, 660, 66, 975, 1565, 1625, 1974, 1043]
                    num_true = [140330, 746524, 566607, 340981, 691685, 218454, 460621, 1003888]
                    world_size = dist.get_world_size()
                    simid = simids[int(dist.get_rank())]
                    start_time = time.time()
                    gen_data = gen(simid, model)
                    end_time = time.time()
                    print(f"Generation time for simid {simid} on rank {dist.get_rank()}: "
                          f"{(end_time - start_time) / 60} mins", flush=True)
                    hmf, ps, num_infer = get_ratios(simid, gen_data, xfirst=True)
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
                            log_ps[r, 1] = ps[1]
                            log_hmf[r, 1] = hmf[1]
                            log_ps[r, 0] = ps[0]
                            log_hmf[r, 0] = hmf[0]
                            print(f"Number of inferred halos: {num_infer}", flush=True)
                            wandb.log(
                                {f"Statistics/NumHalos_GPU{r}_true{num_true[r]}": num_infer},
                                step=sub_step,
                            )
                        base_dir = "/u/yzhang116/NN/plots"
                        plot_dir = os.path.join(base_dir, f"{job_id}")
                        os.makedirs(plot_dir, exist_ok=True)
                        plot_name = os.path.join(plot_dir, f"performance_step_{sub_step}_{job_id}.png")
                        plot_log(log_ps, log_hmf, num_true, plot_name)
                    model.train()

    if dist.get_rank() == 0:
        wandb.finish()
        print("Training finished.")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
