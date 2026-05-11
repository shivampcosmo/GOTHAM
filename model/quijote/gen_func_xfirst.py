import torch
from model_enc_dec_cos_fast_xfirst import *
import numpy as np
import glob
import os
import sys


def gen(simid, model):
    dev = model.device
    nvocab = 131
    start_token = nvocab + 1
    pad_token = nvocab + 3
    end_token = nvocab + 4
    target_len = 296
    nsubox = 64**3
    num_chunks = 16
    chunk_size = (nsubox + num_chunks - 1) // num_chunks
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    all_param = np.loadtxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_params.txt')
    dmo_dir = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3/'
    param = all_param[simid]
    param = np.repeat(param[None,:], nsubox, axis=0)
    params_rep = torch.tensor(param, device=dev).bfloat16()
    dmo = np.load(dmo_dir+f'{simid}/dmo_fields_subvols_grid_8_LH_{simid}.npy')
    dmo = torch.from_numpy(dmo).to(dev).bfloat16()
    dmo = torch.moveaxis(dmo, -1, 1)
    outputs = []
    model.eval()
    with torch.no_grad():
        with ctx:
            for i in range(num_chunks):
                print(f'Generating sim {simid}, chunk {i+1}/{num_chunks}', flush=True)
                s = i * chunk_size
                e = min((i + 1) * chunk_size, nsubox)
                # print(f'Generating sim {simid}, chunk {i+1}/{num_chunks}, samples {s} to {e}', flush=True)
                if s >= e:
                    break
                dmo_chunk = dmo[s:e]
                params_chunk = params_rep[s:e]

                out_chunk = model.module.generate(
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
    return output.cpu().numpy().astype(np.int16)
