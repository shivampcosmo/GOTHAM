import torch
from model_enc_dec_cos import *
import numpy as np
import glob
import os
import sys
from collections import OrderedDict


train_id = int(sys.argv[1])
step = int(sys.argv[2])
lr = sys.argv[4]
title = sys.argv[3]

ckpt_paths = [
f'/work/hdd/bdne/yzhang116/checkpoints_quijote/{train_id}/{title}_cross_entropy_epoch_0_step_{step}_embed_384_batch_800_lrmax_{lr}_lrmin_{lr}_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt'
]

print("Checkpoint paths:")
for path in ckpt_paths:
    print(path)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

avg_state_dict = OrderedDict()
num_ckpts = len(ckpt_paths)

for i, ckpt_path in enumerate(ckpt_paths):
    checkpoint = torch.load(ckpt_path, map_location=dev)
    state_dict = checkpoint["model"]
    print(checkpoint['loss'])

    # 去掉 DataParallel 的 module.
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if i == 0:
        # 初始化
        for k, v in state_dict.items():
            avg_state_dict[k] = v.clone()
    else:
        # 累加
        for k, v in state_dict.items():
            avg_state_dict[k] += v

for k in avg_state_dict:
    avg_state_dict[k] /= num_ckpts

HaloConfig = checkpoint['config']  # 用最后一个或第一个都行

model = HaloDecoderModel(HaloConfig).to(dev)
model.load_state_dict(avg_state_dict)
model.eval()

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
all_param = np.loadtxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_params.txt')
dmo_dir = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3/'
simids = np.loadtxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_test_idx.txt', dtype=int)
files = [dmo_dir + f'{simid}/dmo_fields_subvols_grid_8_LH_{simid}.npy' for simid in simids]

nsubox = 64**3
num_chunks = 16
chunk_size = (nsubox + num_chunks - 1) // num_chunks
rank = int(os.environ.get("SLURM_PROCID", 0))
Ndevices = int(os.environ.get("SLURM_NTASKS", 1))
num_per_device = 30 // Ndevices
start = rank * num_per_device
end = start + num_per_device


nvocab = 131
start_token = nvocab + 1
pad_token = nvocab + 3
end_token = nvocab + 4
target_len = 296
#simids = [1414,660,66,975]

#if 1:
#    simid = simids[rank]
for simid in simids[start:end]:
    param = all_param[simid]
    param = np.repeat(param[None,:], nsubox, axis=0)
    params_rep = torch.tensor(param, device=dev).bfloat16()
    dmo = np.load(dmo_dir+f'{simid}/dmo_fields_subvols_grid_8_LH_{simid}.npy')
    dmo = torch.from_numpy(dmo).to(dev).bfloat16()
    dmo = torch.moveaxis(dmo, -1, 1)
    outputs = []
    with torch.no_grad():
        with ctx:
            for i in range(num_chunks):
                s = i * chunk_size
                e = min((i + 1) * chunk_size, nsubox)
                print(f'Generating sim {simid}, chunk {i+1}/{num_chunks}, samples {s} to {e}', flush=True)
                if s >= e:
                    break
                dmo_chunk = dmo[s:e]
                params_chunk = params_rep[s:e]

                out_chunk = model.generate(
                    dmo_chunk,
                    params=params_chunk,
                    max_new_tokens=289,
                    start_token=start_token,
                    end_token=end_token,
                    pad_token=pad_token,
                )

                # out_chunk: [B, L]
                B, L = out_chunk.shape

                if L < target_len:
                    pad_len = target_len - L
                    pad = torch.full(
                        (B, pad_len),
                        pad_token,
                        dtype=out_chunk.dtype,
                        device=out_chunk.device,
                    )
                    out_chunk = torch.cat([out_chunk, pad], dim=1)

                outputs.append(out_chunk)

                del dmo_chunk
                del params_chunk
                del out_chunk
                torch.cuda.empty_cache()

    output = torch.cat(outputs, dim=0)  
    
    output_np = output.cpu().numpy()
    output_np = np.int16(output_np)
    output_dir = os.path.join('/work/hdd/bdne/yzhang116/generates_quijote', f"cp_{train_id}_step_{step}")
    os.makedirs(output_dir, exist_ok=True)
    out_filename =  os.path.join(output_dir, f'generated_halos_sentence_{simid}.npy')
    np.save(out_filename, output_np)
    print(f'Saved generated halos for sim {simid} to {out_filename}')
