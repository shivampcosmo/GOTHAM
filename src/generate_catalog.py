import torch
import numpy as np
from model_enc_dec_cos import *
import numpy as np
import glob
import os
import sys

ckpt_path = f'/work/hdd/bdne/yzhang116/checkpoints_quijote/??'

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(ckpt_path, map_location=dev)
state_dict = checkpoint["model"]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

HaloConfig = checkpoint['config']

model = HaloDecoderModel(HaloConfig).to(dev)
model.load_state_dict(state_dict)
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
totnum = len(simids)
num_per_device = totnum // Ndevices
start = rank * num_per_device
end = start + num_per_device

nvocab = 131
start_token = nvocab + 1
space_token = nvocab + 2
pad_token = nvocab + 3
end_token = nvocab + 4
target_len = 296
grid = 64
pos_vocab = 40
add_space_token = False
dim_tot = 8
dim_prop = 5

xarray = np.arange(pos_vocab) * (BoxSize / pos_vocab / grid)
xarray = np.concatenate((xarray, [BoxSize/grid]))
dx = BoxSize / (pos_vocab * grid)
bins_digitize = np.zeros((dim_prop, nvocab+1))
bins_digitize[0,:-1] = np.linspace(12.7, 15.0, nvocab)
bins_digitize[0,-1] = 15.0
for i in range(1,4):
    bins_digitize[i,:-1] = np.linspace(-1250, 1250, nvocab)
    bins_digitize[i,-1] = 1250.0
bins_digitize[4,:-1] = np.linspace(1.0, 16.0, nvocab)
bins_digitize[4,-1] = 16.0
bins_step = np.zeros(dim_prop)
bins_step[0] = (15.0 - 12.7) / (nvocab - 1)
for i in range(1,4):
    bins_step[i] = (1250.0 - (-1250.0)) / (nvocab - 1)
bins_step[4] = (16.0 - 1.0) / (nvocab - 1)

def get_prop_pos(X_val):
    pos_infer_all = []
    prop_infer_all = []
    for jx in range(grid):
        for jy in range(grid):
            for jz in range(grid):
                sentence_here = X_val[jx, jy, jz]
                if add_space_token:
                    ntokens_per_halo = dim_tot + 1
                else:
                    ntokens_per_halo = dim_tot
                ind_start_token = np.where(sentence_here == start_token)[0][0]
                
                if end_token in sentence_here:
                    ind_end_token = np.where(sentence_here == end_token)[0][0]
                    Nhalos_here = ((ind_end_token - ind_start_token - 1) / ntokens_per_halo)
                    if (int(Nhalos_here) - Nhalos_here) != 0:
                        print(Nhalos_here, 'Nhalos_here is not an integer')
                    else:
                        if Nhalos_here > 0:
                            for jh in range(int(Nhalos_here)):
                                try:
                                    prop_all = np.zeros(dim_prop, dtype=np.float32)        
                                    for jp in range(dim_prop):
                                        bin_val_jp = sentence_here[ind_start_token + jh*ntokens_per_halo + 4 + jp]
                                        prop_all[jp] = (bins_digitize[jp, bin_val_jp] +np.random.uniform(-0.5,0.5) * bins_step[jp]).clip(min=bins_digitize[jp,0], max=bins_digitize[jp,-1])

                                    coord_x = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 1]] + (BoxSize/grid)*jx + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                    coord_y = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 2]] + (BoxSize/grid)*jy + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                    coord_z = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 3]] + (BoxSize/grid)*jz + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                    pos_infer_all.append([coord_x, coord_y, coord_z])
                                    prop_infer_all.append(prop_all)

                                except Exception as e:
                                   print(e)
                                pass
                else:           
                    print('End token not found')
                    pass
    return np.array(pos_infer_all), np.array(prop_infer_all)

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
    data = output.cpu().numpy()
    data = np.int16(data)
    data = np.delete(data, [1,2,3,4,5,6], axis=1)
    data = data.reshape((grid,grid,grid,-1))
    pos, prop = get_prop_pos(data)
    prop[:,0] = np.power(10.,prop[:,0])
    catalog = np.concatenate([pos, prop], axis=1)
    output_dir = '/work/hdd/bdne/yzhang116/generates_quijote'
    os.makedirs(output_dir, exist_ok=True)
    out_filename =  os.path.join(output_dir, f'generated_halo_catalog_{simid}.npy')
    np.save(out_filename, catalog)
    print(f'Saved generated halos for sim {simid} to {out_filename}')
