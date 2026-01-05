import numpy as np
import sys,os
import MAS_library as MASL
import pickle as pk
import skimage.measure as skmeasure
from mpi4py import MPI
from scipy.ndimage import convolve

###### MPI DEFINITIONS ######
# comm   = MPI.COMM_WORLD
# nprocs = comm.Get_size()
# myrank = comm.Get_rank()

# nstart = 0
# nend   = 1000

# numbers = np.arange(nstart, nend, dtype=int)
# numbers = numbers[numbers % nprocs == myrank]

def fill_empty_voxels_with_local_average(vel_field):
    mask = (vel_field != 0.).astype(np.float32)
    kernel = np.ones((3, 3, 3), dtype=np.float32)
    vel_sum = convolve(vel_field* mask, kernel, mode='wrap')
    weight_sum = convolve(mask, kernel, mode='wrap')
    weight_sum[weight_sum == 0] = 1.0
    vel_avg = vel_sum / weight_sum
    vel_filled= np.where(mask > 0, vel_field, vel_avg)
    
    return vel_filled


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

def process_LH_sim(ji,grid_sbox = 8):
    try:
        grid = 64
        # nrand_sel_box = grid**3
        nrand_sel_box = 32768
        #snaps = ['05','07','1']
        snaps = ['05']

        root_in = '/mnt/ceph/users/spandey/discodj_runs/LH/%d/pm' % ji
        root_out = '/mnt/ceph/users/spandey/discodj_runs/rhog_LH_nsel_%d/%d/' % (nrand_sel_box, ji)
        os.makedirs(root_out, exist_ok=True)

        #savefname_dmo_fields = '/work/hdd/bdne/yzhang116/CV/dmo_fields/dmo_fields_CV_%d.npy'%(sim_id)  
        #input_pos = '/work/hdd/bdne/yzhang116/CV/CV_%d/pm/pos_CV%d_z05.npy'%(sim_id,sim_id)
        #input_vel = '/work/hdd/bdne/yzhang116/CV/CV_%d/pm/vel_CV%d_z05.npy'%(sim_id,sim_id)

        savefname_dmo_fields = root_out + 'dmo_fields_subvols_grid_%d_LH_%d.npy'%(grid_sbox, ji)  
        # savefname_rand_indices = root_out + 'rand_indices_subvols_grid_%d_LH_%d.npy'%(grid_sbox, ji)  
        meta_fname = root_out + 'meta_dmo_fields_subvols_grid_%d_LH_%d.pkl'%(grid_sbox, ji)
        # check if file exists:
        # if os.path.exists(savefname_dmo_fields):
        #     print(f'File exists: {savefname_dmo_fields}')
        #     return
        # else:
        if os.path.exists(savefname_dmo_fields) and os.path.exists(meta_fname):
            print(f'File exists: {savefname_dmo_fields}')
            return
        else:
            print(f'Processing LH sim {ji}')
            for js, snapnum in enumerate(snaps):
                norm_delta = 10
                norm_vel = 100
                BoxSize = 1000.

                MAS_type = 'CIC'
                grid_tot = grid_sbox * grid

                import numpy as np
                np.random.seed(0)
                pos_m_truth = np.load(root_in + '/pos_LH%d_z05.npy' % ji)
                vel_m_truth = np.load(root_in + '/vel_LH%d_z05.npy' % ji)         

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
                    vel_m_jc = fill_empty_voxels_with_local_average(vel_m_jc)            
                    vel_m_part[..., jc] = vel_m_jc/norm_vel

                vel_m_part_rs = mat_reshape(vel_m_part, grid, grid_sbox)

                dmo_fields_all_snap = np.concatenate((Npart_rs[...,None], Npart_pad1_rs[...,None], Npart_pad2_rs[...,None], vel_m_part_rs), axis=-1) # (gird,gird,gird, 8, 8, 8, 6)

                if js == 0:
                    dmo_fields_all = dmo_fields_all_snap
                else:
                    dmo_fields_all = np.concatenate((dmo_fields_all, dmo_fields_all_snap), axis=-1)


            dmo_fields_all_rs = dmo_fields_all.reshape((grid**3, *dmo_fields_all.shape[3:])) # (grid**3, 8, 8, 8, 6)

            if nrand_sel_box < grid**3:
                import numpy as np
                np.random.seed(0)
                # import numpy as np
                # rng = np.random.default_rng(12345)
                dmo_fields_all_rs_z0 = dmo_fields_all_rs[:, ..., 0]
                Npart_sum = np.sum(dmo_fields_all_rs_z0, axis=(1, 2, 3))

                npart_min = np.percentile(Npart_sum, 2.0)
                npart_max = np.percentile(Npart_sum, 98.0)
                
                hist, bins_edges = np.histogram(np.clip(Npart_sum, npart_min, npart_max) , bins=16)
                bins_edges[0] = 0.0
                bins_edges[-1] = bins_edges[-1]*100
                nsel_per_jb = nrand_sel_box//len(hist)

                indsel_all = []
                for jbd in range(len(bins_edges)-1):
                    b1 = bins_edges[jbd]
                    b2 = bins_edges[jbd + 1]
                    indsel = np.where((Npart_sum >= b1) & (Npart_sum < b2))[0]
                    if len(indsel) < nsel_per_jb:
                        ind_in_jb = indsel
                    else:
                        # ind_in_jb = rng.choice(indsel, nsel_per_jb, replace=False)
                        ind_in_jb = np.random.choice(indsel, nsel_per_jb, replace=False)
                    indsel_all.append(ind_in_jb)

                indsel_all = np.concatenate(indsel_all)

                if len(indsel_all) < nrand_sel_box:
                    nmore_to_sel = nrand_sel_box - len(indsel_all)
                    ind_to_sel_remaining = np.setdiff1d(np.arange(grid**3), indsel_all)
                    # indsel_remaining = rng.choice(ind_to_sel_remaining, nmore_to_sel, replace=False)
                    indsel_remaining = np.random.choice(ind_to_sel_remaining, nmore_to_sel, replace=False)
                    indsel_all = np.concatenate((indsel_all, indsel_remaining))

                Npart_sum_sel = Npart_sum[indsel_all]
                # rand_sel = rng.permutation(indsel_all)
                rand_sel = np.random.permutation(indsel_all)
            else:
                import numpy as np
                rand_sel = np.arange(grid**3)
                Npart_sum_sel = 0

            np.save(savefname_dmo_fields, dmo_fields_all_rs.astype(np.float16)[rand_sel, ...])
            # np.save(savefname_rand_indices, rand_sel)
            saved = {
                    'snapnums': snaps,
                    'Npart_sum_all': Npart_sum,
                    'Npart_sum_sel': Npart_sum_sel,
                    'norm_delta': norm_delta,
                    'norm_vel': norm_vel,
                    'grid_sbox':grid_sbox,
                    'grid':grid,
                    'rand_sel': rand_sel,
                    'MAS_type':MAS_type         
                    }
            pk.dump(saved, open(meta_fname, 'wb'))
            return
    
    except Exception as e:
        print(f'Error: {e}')
        return

# if __name__ == '__main__':
#     for sim_id in numbers:
#         process_LH_sim(sim_id)

sim_id = int(sys.argv[1])
process_LH_sim(sim_id)

# import multiprocessing as mp
# if __name__ == '__main__':
#     n_sims = 100
#     n_sims_offset = int(sys.argv[1])
#     # n_sims = 400

#     # n_sims_offset = int(sys.argv[-1])
#     # n_sims = int(sys.argv[-2])
#     print(n_sims_offset, n_sims)

#     nthreads_total = mp.cpu_count()
#     n_procs = 10
#     nthreads_per_proc = nthreads_total // n_procs
#     print(n_procs)

#     # Create a pool of worker processes
#     pool = mp.Pool(processes=n_procs)

#     # Distribute the simulations across the available cores
#     sims_per_core = n_sims // n_procs
#     sim_ranges = [(n_sims_offset + i * sims_per_core, n_sims_offset + (i + 1) * sims_per_core) for i in range(n_procs)]

#     print(sims_per_core, sim_ranges)

#     # Handle any remaining simulations
#     remaining_sims = n_sims % n_procs
#     if remaining_sims > 0:
#         sim_ranges[-1] = (sim_ranges[-1][0], sim_ranges[-1][1] + remaining_sims)

#     # Run save_cic_densities function for each simulation range in parallel
#     results = [pool.apply_async(process_LH_sim, args=(ji,)) for sim_range in sim_ranges for ji in range(*sim_range)]

#     # Wait for all tasks to complete
#     [result.get() for result in results]

#     # Close the pool and wait for tasks to finish
#     pool.close()
#     pool.join()
