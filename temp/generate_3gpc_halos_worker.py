"""
Worker script for generating 3Gpc halo catalogs.
Each worker processes a subset of chunks on a single GPU.

Usage:
    python generate_3gpc_halos_worker.py --simid 663 --worker_id 0 --num_workers 8
"""
import sys
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import gc
sys.path.append('/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/src')
from model_enc_dec_cos import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simid', type=int, required=True)
    parser.add_argument('--worker_id', type=int, required=True, help='Worker index (0-based)')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    simid = args.simid
    worker_id = args.worker_id
    num_workers = args.num_workers

    output_dir = '/mnt/ceph/users/spandey/discodj_runs/test_3gpc/gen_cats'
    partial_dir = os.path.join(output_dir, f'partial_{simid}')
    os.makedirs(partial_dir, exist_ok=True)

    # Model setup
    ckpt_path = '/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/checkpoints/checkpoint_new.pt'
    dev = torch.device('cuda')

    checkpoint = torch.load(ckpt_path, map_location=dev)
    state_dict = checkpoint["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    HaloConfig = checkpoint['config']
    model = HaloDecoderModel(HaloConfig).to(dev)
    model.load_state_dict(state_dict)
    model.eval()

    for i in range(len(model.transformer.h)):
        model.transformer.h[i] = torch.compile(model.transformer.h[i], mode="default")

    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # Data setup
    all_param = np.loadtxt('/mnt/ceph/users/spandey/quijote_v2_gotham/GOTHAM/checkpoints/quijote_params.txt')
    dmo_dir = '/mnt/ceph/users/spandey/discodj_runs/test_3gpc/full_rhog_LH/'

    BoxSize = 3000.0
    grid = 3 * 64
    nsubox = grid ** 3
    num_chunks = (3 * 4) ** 3 // 3
    chunk_size = (nsubox + num_chunks - 1) // num_chunks
    target_len = 296

    nvocab = 131
    start_token = nvocab + 1
    end_token = nvocab + 4
    pad_token = nvocab + 3

    # Divide chunks across workers
    chunks_per_worker = (num_chunks + num_workers - 1) // num_workers
    chunk_start = worker_id * chunks_per_worker
    chunk_end = min((worker_id + 1) * chunks_per_worker, num_chunks)

    print(f'Worker {worker_id}/{num_workers}: processing chunks {chunk_start} to {chunk_end-1} '
          f'({chunk_end - chunk_start} chunks out of {num_chunks} total)', flush=True)

    # Load data
    param = all_param[simid]
    param = np.repeat(param[None, :], nsubox, axis=0)
    params_rep = torch.tensor(param, device=dev).bfloat16()

    dmo = np.load(os.path.join(dmo_dir, f'{simid}/dmo_fields_subvols_grid_8_LH_{simid}.npy'))
    dmo = torch.from_numpy(dmo)
    dmo = torch.moveaxis(dmo, -1, 1)

    # Process assigned chunks
    outputs = []
    with torch.no_grad():
        with ctx:
            for i in tqdm(range(chunk_start, chunk_end), desc=f'Worker {worker_id}'):
                s = i * chunk_size
                e = min((i + 1) * chunk_size, nsubox)
                if s >= e:
                    break
                print(f'Worker {worker_id}: chunk {i+1}/{num_chunks}, samples {s} to {e}', flush=True)

                dmo_chunk = dmo[s:e].to(dev).bfloat16()
                params_chunk = params_rep[s:e]

                out_chunk = model.generate(
                    dmo_chunk,
                    params=params_chunk,
                    max_new_tokens=289,
                    start_token=start_token,
                    end_token=end_token,
                    pad_token=pad_token,
                )

                B, L = out_chunk.shape
                if L < target_len:
                    pad_len = target_len - L
                    pad = torch.full(
                        (B, pad_len), pad_token,
                        dtype=out_chunk.dtype, device=out_chunk.device,
                    )
                    out_chunk = torch.cat([out_chunk, pad], dim=1)

                outputs.append(out_chunk.cpu())
                del dmo_chunk, params_chunk, out_chunk
                gc.collect()
    
    torch.cuda.empty_cache()

    output = torch.cat(outputs, dim=0)
    data = output.numpy().astype(np.int16)

    # Save partial output
    out_path = os.path.join(partial_dir, f'partial_{worker_id}.npy')
    np.save(out_path, data)
    print(f'Worker {worker_id}: saved partial output to {out_path} with shape {data.shape}', flush=True)


if __name__ == '__main__':
    main()
