import torch
import numpy as np
from model_enc_dec_cos_fast import *
import glob
import os
import sys
import math
import time

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.preferred_blas_library("cublaslt")

ckpt_path = f'/work/hdd/bdne/yzhang116/checkpoints_quijote/1872236/checkpoint_cos_cross_entropy_epoch_0_step_7050_embed_384_batch_800_lrmax_1e-05_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt'

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(ckpt_path, map_location=dev)
state_dict = checkpoint["model"]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

HaloConfig = checkpoint['config']

model = HaloDecoderModel(HaloConfig).to(dev)
model.load_state_dict(state_dict)
model.eval()

for i in range(len(model.transformer.h)):
    model.transformer.h[i] = torch.compile(model.transformer.h[i], mode='default')

print("Model loaded and compiled for inference.", flush=True)
print(dev, flush=True)

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
all_param = np.loadtxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_params.txt')
dmo_dir = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3/'

BoxSize = 1000.0
nsubox = 64**3
num_chunks = 16
chunk_size = (nsubox + num_chunks - 1) // num_chunks

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
                                        prop_all[jp] = (bins_digitize[jp, bin_val_jp] + np.random.uniform(-0.5,0.5) * bins_step[jp]).clip(min=bins_digitize[jp,0], max=bins_digitize[jp,-1])

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


def run_inference(dmo_flat, params_rep, simid, seed_offset):
    outputs = []
    with torch.no_grad():
        with ctx:
            for i in range(num_chunks):
                s = i * chunk_size
                e = min((i + 1) * chunk_size, nsubox)
                if s >= e:
                    break
                dmo_chunk = dmo_flat[s:e]
                params_chunk = params_rep[s:e]

                torch.manual_seed(seed_offset * num_chunks + i)
                torch.cuda.manual_seed_all(seed_offset * num_chunks + i)
                out_chunk = model.generate(
                    dmo_chunk,
                    params=params_chunk,
                    max_new_tokens=289,
                    start_token=start_token,
                    end_token=end_token,
                    pad_token=pad_token,
                    #temperature=0.5,
                )

                B, L = out_chunk.shape
                if L < target_len:
                    pad_len = target_len - L
                    pad = torch.full((B, pad_len), pad_token, dtype=out_chunk.dtype, device=out_chunk.device)
                    out_chunk = torch.cat([out_chunk, pad], dim=1)

                outputs.append(out_chunk)
                del dmo_chunk, params_chunk, out_chunk

    return torch.cat(outputs, dim=0)


# Velocity channel groups for each spatial axis: vx=[7,15], vy=[8,16], vz=[9,17]
VEL_CHANNEL_INDICES = [[7, 15], [8, 16], [9, 17]]

# 6 permutations of (x, y, z) spatial axes
spatial_perms = [
    ((0, 1, 2), 'xyz'),
    ((0, 2, 1), 'xzy'),
    ((1, 0, 2), 'yxz'),
    ((1, 2, 0), 'yzx'),
    ((2, 0, 1), 'zxy'),
    ((2, 1, 0), 'zyx'),
]

simid = 66
param = all_param[simid]
param = np.repeat(param[None,:], nsubox, axis=0)
params_rep = torch.tensor(param, device=dev).bfloat16()

# Load dmo and move channel axis: raw shape -> (B, C, 8, 8, 8)
dmo = np.load(dmo_dir + f'{simid}/dmo_fields_subvols_grid_8_LH_{simid}.npy')
dmo = torch.from_numpy(dmo).to(dev).bfloat16()
dmo = torch.moveaxis(dmo, -1, 1)  # (B, C, Dx, Dy, Dz)

C = dmo.shape[1]
# Reshape macro subvolume grid: (64, 64, 64, C, 8, 8, 8)
dmo_grid = dmo.reshape(grid, grid, grid, C, 8, 8, 8)

output_dir = f'/work/hdd/bdne/yzhang116/generate_catalog_quijote/permuted'
os.makedirs(output_dir, exist_ok=True)

for perm, pname in spatial_perms:
    p0, p1, p2 = perm
    start_time = time.time()
    print(f'Processing permutation {pname} for sim {simid}...', flush=True)

    # Permute both macro (subvol grid) and micro (internal voxel) spatial dims.
    # dmo_grid shape: (64_x, 64_y, 64_z, C, 8_x, 8_y, 8_z)
    # After permutation (p0,p1,p2): macro dims [p0,p1,p2], C stays at 3, micro dims [4+p0,4+p1,4+p2]
    dmo_perm = dmo_grid.permute(p0, p1, p2, 3, 4+p0, 4+p1, 4+p2).contiguous()
    dmo_flat = dmo_perm.reshape(nsubox, C, 8, 8, 8)

    # Remap velocity channels so new-axis-i carries velocity along original axis perm[i].
    # e.g. for perm (1,0,2): new vx channels [7,15] should hold old vy values [8,16].
    channel_map = list(range(C))
    for new_ax, old_ax in enumerate(perm):
        for new_ch, old_ch in zip(VEL_CHANNEL_INDICES[new_ax], VEL_CHANNEL_INDICES[old_ax]):
            channel_map[new_ch] = old_ch
    dmo_flat = dmo_flat[:, channel_map, :, :, :]

    # Use a seed offset based on permutation index to get independent samples
    perm_idx = [p for p, _ in spatial_perms].index(perm)
    seed_offset = simid * 10 + perm_idx

    output = run_inference(dmo_flat, params_rep, simid, seed_offset)
    data = output.cpu().numpy()
    data = np.int16(data)
    data = np.delete(data, [1,2,3,4,5,6], axis=1)
    data = data.reshape((grid, grid, grid, -1))

    np.random.seed(seed_offset)
    pos, prop = get_prop_pos(data)
    prop[:,0] = np.power(10., prop[:,0])
    catalog = np.concatenate([pos, prop], axis=1)

    # Inverse-permute positions and velocities back to original (x,y,z) frame.
    # perm (p0,p1,p2) means output axis i holds coordinates along original axis pi.
    # inv_perm[pi] = i, so original[:, j] = catalog[:, inv_perm[j]].
    # catalog columns: [x, y, z, mass, vx, vy, vz, prop4]
    inv_perm = [0, 0, 0]
    for i, pi in enumerate(perm):
        inv_perm[pi] = i
    catalog[:, 0:3] = catalog[:, inv_perm]          # positions
    catalog[:, 4:7] = catalog[:, [4 + j for j in inv_perm]]  # velocities

    out_filename = os.path.join(output_dir, f'generated_halo_catalog_{simid}_perm_{pname}.npy')
    np.save(out_filename, catalog)

    end_time = time.time()
    print(f'Saved permutation {pname}: {len(catalog)} halos, time {(end_time-start_time)/60:.2f} mins -> {out_filename}', flush=True)
