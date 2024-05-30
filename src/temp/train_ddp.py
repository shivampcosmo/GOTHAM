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
dev = torch.device("cuda")
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
# ------------
# vocab_size = 131
# block_size = 161

vocab_size = 67
block_size = 211


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


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break


# def run_fn(rank, world_size):
def run_fn():    
    # world_size    = int(os.environ["WORLD_SIZE"])
    # rank          = int(os.environ["SLURM_PROCID"])
    # gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # # assert gpus_per_node == torch.cuda.device_count()
    # print(f"Hello from rank {rank} of {world_size} on FI where there are" \
    #       f" {gpus_per_node} allocated GPUs per node. Total devices are {torch.cuda.device_count()}", flush=True)
    # assert gpus_per_node == torch.cuda.device_count()
    # setup(rank, world_size)
    # if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    # local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    # print(f"host: FI, rank: {rank}, local_rank: {local_rank}, dev name: {torch.cuda.current_device()}")
    # torch.cuda.set_device(local_rank)

    # Initialize process group
    # dist.init_process_group(backend='nccl',rank=rank, world_size=world_size)
    # device = local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # NCCL is the protocol that should be used to communicate between GPUs
    # torch.distributed.init_process_group("nccl")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    device = rank % torch.cuda.device_count()
    # torch.cuda.set_device(device)
    # device = rank 
    print(f"host: FI, rank: {rank}, local_rank: {rank}, dev name: {torch.cuda.current_device()}")
    # Set up random seeds
    # torch.manual_seed(0)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    device_type = 'cuda' 
    # note: float16 data type will automatically use a GradScaler
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if rank == 0: print(f"Loading data", flush=True)    
    f = h5.File('/mnt/home/spandey/ceph/CHARFORMER/data/PM/df_halo_part_ngp_xyzM_tokenized_PM_density3Dgrid_32_isim_012_snap_3.h5', 'r')
    dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all = f['dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all'][:]
    delta_box_all_squeezed_all = f['delta_box_all_squeezed_all'][:]
    f.close()
    if rank == 0: print(f"Loaded data", flush=True)        

    x_train, y_train, dm_train, mask_train, x_val, y_val, dm_val, mask_val = get_data_split(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_all, delta_box_all_squeezed_all, 0.1, 0.2)
    if rank == 0: print(f"Got split with sizes {x_train.shape} and {x_val.shape}", flush=True)        

    start_token = 0
    pad_token = 1
    end_token = int(torch.max(x_train).cpu().numpy()) - 1
    space_token = int(torch.max(x_train).cpu().numpy())


    # Define model
    # model = torchvision.models.resnet18()
    
    # Wrap model with DistributedDataParallel
    # model = DistributedDataParallel(model)
    
    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Load data
    train_dataset = CustomDataset(x_train, dm_train, mask_train, y_train)
    train_dataset.set_rank_and_world_size(rank, world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=train_sampler)

    val_dataset = CustomDataset(x_val, dm_val, mask_val, y_val)
    val_dataset.set_rank_and_world_size(rank, world_size)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, sampler=val_sampler)

    x_train_gpu, dm_train_gpu, mask_train_gpu, y_train_gpu = [], [], [], []
    for x, dm, mask, y in train_loader:
        x_train_gpu.append(x.cuda(non_blocking=True))
        dm_train_gpu.append(dm.cuda(non_blocking=True))
        mask_train_gpu.append(mask.cuda(non_blocking=True))
        y_train_gpu.append(y.cuda(non_blocking=True))
    
    x_train_gpu = torch.cat(x_train_gpu, dim=0)
    dm_train_gpu = torch.cat(dm_train_gpu, dim=0)
    mask_train_gpu = torch.cat(mask_train_gpu, dim=0)
    y_train_gpu = torch.cat(y_train_gpu, dim=0)

    x_val_gpu, dm_val_gpu, mask_val_gpu, y_val_gpu = [], [], [], []
    for x, dm, mask, y in val_loader:
        x_val_gpu.append(x.cuda(non_blocking=True))
        dm_val_gpu.append(dm.cuda(non_blocking=True))
        mask_val_gpu.append(mask.cuda(non_blocking=True))
        y_val_gpu.append(y.cuda(non_blocking=True))
    
    x_val_gpu = torch.cat(x_val_gpu, dim=0)
    dm_val_gpu = torch.cat(dm_val_gpu, dim=0)
    mask_val_gpu = torch.cat(mask_val_gpu, dim=0)
    y_val_gpu = torch.cat(y_val_gpu, dim=0)

    model = DDP(HaloDecoderModel(HaloConfig), device_ids=[device])
    # model.to(device).bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
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
            x = x[batch_size*(ji):batch_size*(ji+1)].to(device, non_blocking=True)
            y = y[batch_size*(ji):batch_size*(ji+1)].to(device, non_blocking=True)
            mask = mask[batch_size*(ji):batch_size*(ji+1)].to(device, non_blocking=True)
            dm = dm[batch_size*(ji):batch_size*(ji+1)].to(device, non_blocking=True)

        return x, y, mask, dm

    # helps estimate an arbitrarily accurate loss over either split using many batches
    def estimate_loss():
        out = {}
        # model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, MASK, DM = get_batch(split, batch_size = batch_size)
                with ctx:
                    logits, loss = model(X, DM, maskd=MASK, targets=Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out

    print("compiling the model... (takes a ~minute)")
    # unoptimized_model = model
    # model = torch.compile(model) # requires PyTorch 2.0

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # model.load_state_dict(torch.load('/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec_PM.pt'))

    # Training loop
    val_loss_min = 1e20
    # saved = {}
    eval_interval = 40
    nbatches = 24
    saved = {}
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            with torch.no_grad():
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                # save the model if it's better than the previous best:
                if losses['val'] < val_loss_min:
                    val_loss_min = losses['val']
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'best_loss': val_loss_min,
                        'epoch': iter
                    }
                    if rank == 0:
                        # torch.save(model.state_dict(), '/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec.pt')
                        torch.save(checkpoint, '/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec_ddp_PM_isim_012_nvocab_64_nembed_64.pt')            
                        print(f"New best model saved with loss {val_loss_min:.4f}")
                    dist.barrier()

        for ji in (range(nbatches)):
            X, Y, MASK, DM = get_batch('train', ji, batch_size)
            # with ctx:
            _, loss = model(X, DM, maskd=MASK, targets=Y)
                # loss = loss
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        if iter % 10 == 0:
            print(f"iter {iter}, loss: {loss.item()}")
        # optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        # optimizer.step()



    
    # Cleanup
    dist.destroy_process_group()



if __name__ == '__main__':
    run_fn()
    # world_size = torch.cuda.device_count()  # Use all available GPUs
    # print(world_size)
# 
    # mp.spawn(run_fn, args=(world_size, ), nprocs=world_size, join=True)

