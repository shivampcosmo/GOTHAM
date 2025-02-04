import numpy as np
import sys,os
import MAS_library as MASL
import pickle as pk
import h5py as h5
from ngp_funcs import NGP_xyz_prop 
import skimage.measure as skmeasure


def mat_reshape(mat, grid1, grid2):
    '''
    Reshape input mat into (grid1, grid1, grid1, grid2, grid2, grid2) shape
    '''
    if len(mat.shape) == 3:
        mat_rs = np.reshape(mat, (grid1, grid2, grid1, grid2, grid1, grid2))
    else:
        mat_rs = np.reshape(mat, (grid1, grid2, grid1, grid2, grid1, grid2, *mat.shape[3:]))

    mat_rs = np.moveaxis(mat_rs, 1, 2)
    mat_rs = np.moveaxis(mat_rs, 4, 3)
    mat_rs = np.moveaxis(mat_rs, 3, 2)
    return mat_rs


def get_padded_mat(Npart, n_pad, grid_sbox, grid):
    Npart_pad = np.pad(Npart, n_pad, 'wrap')

    Npart_pad1 = np.zeros((grid, grid, grid, grid_sbox + 2*n_pad, grid_sbox + 2*n_pad, grid_sbox + 2*n_pad))
    fac = (grid_sbox + 2*n_pad)//grid_sbox    
    Npart_pad1_reduce = np.zeros((grid, grid, grid, grid_sbox, grid_sbox, grid_sbox))    
    xstart, ystart, zstart = n_pad, n_pad, n_pad
    for jx in range(grid):
        for jy in range(grid):
            for jz in range(grid):
                Npart_pad1[jx, jy, jz] = Npart_pad[xstart + jx * grid_sbox - n_pad:xstart + (jx + 1) * grid_sbox + n_pad,
                                                        ystart + jy * grid_sbox - n_pad:ystart + (jy + 1) * grid_sbox + n_pad,
                                                        zstart + jz * grid_sbox - n_pad:zstart + (jz + 1) * grid_sbox + n_pad]

                Npart_pad1_reduce[jx, jy, jz] = skmeasure.block_reduce(Npart_pad1[jx, jy, jz], (fac, fac, fac), np.mean)
    return Npart_pad1_reduce, Npart_pad1

def process_LH_sim(isim_fid):
    nrand_sel_box = 128
    norm_delta = 500
    norm_vel = 500
    BoxSize = 25.
    grid = 8
    grid_sbox = 32
    MAS_type = 'NGP'
    grid_tot = grid_sbox * grid
    snapnums = [90, 84, 78, 70, 60]
    
    
    import numpy as np
    np.random.seed(0)
    rand_sel = np.sort(np.random.randint(0, grid**3, nrand_sel_box)).astype(int)
    
    sdir = '/work/hdd/bdne/spandey3/camels_tng/gotham_data/LH/DMO_fields'
    savefname_dmo_fields = f'{sdir}/DMO_fields_grid_{grid_sbox}_isim_{isim_fid}_nrandsubsel_{nrand_sel_box}_MAS_{MAS_type}_nsnaps_{len(snapnums)}.pkl'
    
    
    for js, snapnum in enumerate(snapnums):
        root = '/work/hdd/bdne/spandey3/camels_tng/DMO/LH'
        snap_fname =  f'{root}/LH_{isim_fid}/snapshot_0{snapnum}.hdf5'
    
        np.random.seed(0)
        df = h5.File(snap_fname, 'r')                
        pos_m_truth = df['PartType1']['Coordinates'][()]/1000.
        vel_m_truth = df['PartType1']['Velocities'][()]
        ids_m_truth = np.arange(len(pos_m_truth))
    
        rho_bar = len(pos_m_truth)/(BoxSize**3)
        vol_vox = (BoxSize/grid_tot)**3
        N_bar_vox = rho_bar * vol_vox
    
        Npart = np.float32(np.zeros((grid_tot, grid_tot, grid_tot)))
        MASL.MA(np.float32(pos_m_truth), Npart, BoxSize, MAS_type, verbose=False)
    
        Npart /= N_bar_vox
        Npart /= norm_delta
    
        Npart_rs = mat_reshape(Npart, grid, grid_sbox)
    
        npad1 = grid_sbox
        Npart_pad1_rs, _ = get_padded_mat(Npart, int(npad1), grid_sbox, grid)
    
        npad2 = 2*grid_sbox
        Npart_pad2_rs, _ = get_padded_mat(Npart, int(npad2), grid_sbox, grid)
    
        vel_m_part = np.float32(np.zeros((grid_tot, grid_tot, grid_tot, 3)))
    
        Npart_cic = np.float32(np.zeros((grid_tot, grid_tot, grid_tot)))
        MASL.MA(np.float32(pos_m_truth), Npart_cic, BoxSize, 'CIC', verbose=False)
        for jc in range(3):
            mom_jc = np.float32(np.zeros((grid_tot, grid_tot, grid_tot)))
            MASL.MA(np.float32(pos_m_truth), mom_jc, BoxSize, 'CIC', verbose=False, W=vel_m_truth[:,jc].astype(np.float32))
            vel_m_jc = mom_jc/Npart_cic
            vel_m_jc[~np.isfinite(vel_m_jc)] = 0.0
            vel_m_part[..., jc] = vel_m_jc/norm_vel
    
        vel_m_part_rs = mat_reshape(vel_m_part, grid, grid_sbox)
    
        dmo_fields_all_snap = np.concat((Npart_rs[...,None], Npart_pad1_rs[...,None], Npart_pad2_rs[...,None], vel_m_part_rs), axis=-1)
    
        if js == 0:
            dmo_fields_all = dmo_fields_all_snap
        else:
            dmo_fields_all = np.concatenate((dmo_fields_all, dmo_fields_all_snap), axis=-1)
    
    dmo_fields_all_rs = dmo_fields_all.reshape((grid**3, *dmo_fields_all.shape[3:]))
    
    saved = {'dmo_fields_all': dmo_fields_all_rs.astype(np.float32)[rand_sel, ...],
             'snapnums': snapnums,
            'rand_sel': rand_sel,
            'norm_delta': norm_delta,
            'norm_vel': norm_vel,
             'grid_sbox':grid_sbox,
             'grid':grid,
             'MAS_type':MAS_type         
            }
    pk.dump(saved, open(savefname_dmo_fields, 'wb'))


import multiprocessing as mp
if __name__ == '__main__':
    # n_sims = 1100
    n_sims_offset = 0
    n_sims = 1000
    # n_cores = mp.cpu_count()
    n_cores = 4
    print(n_cores)

    # Create a pool of worker processes
    pool = mp.Pool(processes=n_cores)

    # Distribute the simulations across the available cores
    sims_per_core = n_sims // n_cores
    sim_ranges = [(n_sims_offset + i * sims_per_core, n_sims_offset + (i + 1) * sims_per_core) for i in range(n_cores)]

    print(sims_per_core, sim_ranges)

    # Handle any remaining simulations
    remaining_sims = n_sims % n_cores
    if remaining_sims > 0:
        sim_ranges[-1] = (sim_ranges[-1][0], sim_ranges[-1][1] + remaining_sims)

    # Run save_cic_densities function for each simulation range in parallel
    results = [pool.apply_async(process_LH_sim, args=(ji,)) for sim_range in sim_ranges for ji in range(*sim_range)]

    # Wait for all tasks to complete
    [result.get() for result in results]

    # Close the pool and wait for tasks to finish
    pool.close()
    pool.join()
