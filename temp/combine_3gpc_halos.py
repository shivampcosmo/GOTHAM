"""
Combine partial outputs from worker scripts into a final halo catalog.

Usage:
    python combine_3gpc_halos.py --simid 663 --num_workers 8
"""
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm


def get_prop_pos(X_val, grid, BoxSize, nvocab, start_token, end_token,
                 dim_tot, dim_prop, add_space_token, pos_vocab,
                 xarray, dx, bins_digitize, bins_step):
    pos_infer_all = []
    prop_infer_all = []
    if add_space_token:
        ntokens_per_halo = dim_tot + 1
    else:
        ntokens_per_halo = dim_tot

    for jx in tqdm(range(grid)):
        for jy in range(grid):
            for jz in range(grid):
                sentence_here = X_val[jx, jy, jz]
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
                                        bin_val_jp = sentence_here[ind_start_token + jh * ntokens_per_halo + 4 + jp]
                                        prop_all[jp] = (
                                            bins_digitize[jp, bin_val_jp]
                                            + np.random.uniform(-0.5, 0.5) * bins_step[jp]
                                        ).clip(min=bins_digitize[jp, 0], max=bins_digitize[jp, -1])

                                    coord_x = (xarray[sentence_here[ind_start_token + jh * ntokens_per_halo + 1]]
                                               + (BoxSize / grid) * jx
                                               + np.random.uniform(-0.5, 0.5) * dx) % BoxSize
                                    coord_y = (xarray[sentence_here[ind_start_token + jh * ntokens_per_halo + 2]]
                                               + (BoxSize / grid) * jy
                                               + np.random.uniform(-0.5, 0.5) * dx) % BoxSize
                                    coord_z = (xarray[sentence_here[ind_start_token + jh * ntokens_per_halo + 3]]
                                               + (BoxSize / grid) * jz
                                               + np.random.uniform(-0.5, 0.5) * dx) % BoxSize
                                    pos_infer_all.append([coord_x, coord_y, coord_z])
                                    prop_infer_all.append(prop_all)
                                except Exception as e:
                                    print(e)
                                    pass
                else:
                    print('End token not found')
                    pass
    return np.array(pos_infer_all), np.array(prop_infer_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simid', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    simid = args.simid
    num_workers = args.num_workers

    output_dir = '/mnt/ceph/users/spandey/discodj_runs/test_3gpc/gen_cats'
    partial_dir = os.path.join(output_dir, f'partial_{simid}')
    out_filename = os.path.join(output_dir, f'generated_halo_catalog_{simid}.npy')

    # Constants (must match worker script)
    BoxSize = 3000.0
    grid = 3 * 64
    nvocab = 131
    start_token = nvocab + 1
    end_token = nvocab + 4
    pad_token = nvocab + 3
    pos_vocab = 40
    add_space_token = False
    dim_tot = 8
    dim_prop = 5

    xarray = np.arange(pos_vocab) * (BoxSize / pos_vocab / grid)
    xarray = np.concatenate((xarray, [BoxSize / grid]))
    dx = BoxSize / (pos_vocab * grid)

    bins_digitize = np.zeros((dim_prop, nvocab + 1))
    bins_digitize[0, :-1] = np.linspace(12.7, 15.0, nvocab)
    bins_digitize[0, -1] = 15.0
    for i in range(1, 4):
        bins_digitize[i, :-1] = np.linspace(-1250, 1250, nvocab)
        bins_digitize[i, -1] = 1250.0
    bins_digitize[4, :-1] = np.linspace(1.0, 16.0, nvocab)
    bins_digitize[4, -1] = 16.0

    bins_step = np.zeros(dim_prop)
    bins_step[0] = (15.0 - 12.7) / (nvocab - 1)
    for i in range(1, 4):
        bins_step[i] = (1250.0 - (-1250.0)) / (nvocab - 1)
    bins_step[4] = (16.0 - 1.0) / (nvocab - 1)

    # Load and concatenate partial outputs (in order)
    partials = []
    for w in range(num_workers):
        path = os.path.join(partial_dir, f'partial_{w}.npy')
        print(f'Loading {path}')
        partials.append(np.load(path))

    data = np.concatenate(partials, axis=0)
    print(f'Combined data shape: {data.shape}')

    # Remove columns and reshape (same as notebook)
    data = np.delete(data, [1, 2, 3, 4, 5, 6], axis=1)
    data = data.reshape((grid, grid, grid, -1))

    # Decode to catalog
    print('Decoding halo catalog...')
    pos, prop = get_prop_pos(
        data, grid, BoxSize, nvocab, start_token, end_token,
        dim_tot, dim_prop, add_space_token, pos_vocab,
        xarray, dx, bins_digitize, bins_step,
    )
    prop[:, 0] = np.power(10., prop[:, 0])
    catalog = np.concatenate([pos, prop], axis=1)
    np.save(out_filename, catalog)
    print(f'Saved final catalog to {out_filename} with shape {catalog.shape}')

    # Cleanup partial files
    import shutil
    shutil.rmtree(partial_dir)
    print(f'Cleaned up partial directory: {partial_dir}')


if __name__ == '__main__':
    main()
