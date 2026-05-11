import torch
from model_enc_dec_pos import *
import numpy as np
import glob
import os
import sys

def get_sim_number(filepath):
    filename = os.path.basename(filepath)
    number_part = filename.split('_')[-1]
    number = int(number_part.split('.')[0])
    return number

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('/work/hdd/bdne/yzhang116/checkpoints_quijote/checkpoint_pos_cross_entropy_epoch_1_step_45000_embed_256_batch_1600_lrmax_0.0002_lrmin_2e-05_layer_4_head_8_layervit_4_headvit_8_patch_4_dropout_0.0.pt', map_location=dev)

HaloConfig = checkpoint['config']

print(checkpoint['loss'])
state_dict = checkpoint["model"]
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v
model = HaloDecoderModel(HaloConfig).to(dev)
model.load_state_dict(new_state_dict)
model.eval()

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
all_param = np.loadtxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_params.txt')
dmo_dir = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3/'
simids = np.loadtxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_test_idx.txt', dtype=int)
files = [dmo_dir + f'{simid}/dmo_fields_subvols_grid_8_LH_{simid}.npy' for simid in simids]

nsubox = 64**3
num_chunks = 10
chunk_size = (nsubox + num_chunks - 1) // num_chunks
Ndevices = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
num_per_device = 20 // Ndevices
start = rank * num_per_device
end = start + num_per_device

nvocab = 40
start_token = nvocab + 1
pad_token = nvocab + 2
end_token = nvocab + 3
target_len = 116

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
                    max_new_tokens=117,
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
    out_filename = f'/work/hdd/bdne/yzhang116/generates_quijote/generated_halos_sentence_{simid}_pos_256.npy'
    np.save(out_filename, output_np)
    print(f'Saved generated halos for sim {simid} to {out_filename}')
