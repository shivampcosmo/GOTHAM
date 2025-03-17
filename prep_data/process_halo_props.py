import numpy as np
import sys,os
import MAS_library as MASL
import pickle as pk
import h5py as h5
from ngp_funcs import NGP_xyz_prop 
import skimage.measure as skmeasure
import ast

# add_space_token = bool(ast.literal_eval(sys.argv[-1]))

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




def process_LH_sim(isim, isfid=False, LH_cosmo_val_file='/mnt/home/spandey/ceph/Quijote/latin_hypercube_params.txt'):
    import numpy as np
    import sys,os
    import MAS_library as MASL
    import pickle as pk
    import h5py as h5
    from ngp_funcs import NGP_xyz_prop 
    import skimage.measure as skmeasure
    import colossus  
    from colossus.cosmology import cosmology
    from colossus.halo import mass_so      
    import gc
    nrand_sel_box = 8192
    add_space_token = False
    # nrand_sel_box = 512 
    BoxSize = 1000.
    grid = 32
    nvocab = 64
    # Mstar_cut = 8.75
    Mstar_cut = 12.7
    grid_sbox = nvocab
    grid_tot = grid_sbox * grid
    vel_norm = 2000.
    prop_min = np.array([12.7, -0.6, -0.6, -0.6, 0.5])
    prop_max = np.array([15.7, 0.6, 0.6, 0.6, 15.0])

    Npoints_max_per_subvol = 60
    #sort by Mstar token:
    ind_token_to_sort = 3
    # add_space_token = False
    # add_space_token = True

    dim_pos = 3
    dim_prop = len(prop_min)
    dim_tot = dim_pos + dim_prop

    start_token = nvocab + 1
    space_token = nvocab + 2
    pad_token = nvocab + 3
    end_token = nvocab + 4

    # import numpy as np
    # np.random.seed(0)
    # rand_sel = np.sort(np.random.randint(0, grid**3, nrand_sel_box)).astype(int)
    # rand_sel = (np.arange(grid**3)[:nrand_sel_box]).astype(int)
    if nrand_sel_box < grid**3:
        ldir = '/mnt/home/spandey/ceph/Quijote/halo_gotham_data/LH/DMO_fields_ng32'
        savefname_dmo_fields = f'{ldir}/DMO_fields_grid_8_isim_{isim}_nrandsubsel_{nrand_sel_box}_MAS_CIC_nsnaps_2.pkl'
        df = pk.load(open(savefname_dmo_fields,'rb'))
        rand_sel = df['rand_sel']
    else:
        rand_sel = np.arange(grid**3)
        
    sdir = '/mnt/home/spandey/ceph/Quijote/halo_gotham_data/LH'
    snapnum_hres = 73
    savefname_gals = f'{sdir}/halo_props_ng32/halo_props_snap_{snapnum_hres}_grid_{grid_sbox}_isim_{isim}_nrandsubsel_{nrand_sel_box}_nvocab{nvocab}_spacetoken_{add_space_token}_xMvc_{Mstar_cut}.pkl'

    snapnum_hres_dict = {90:4, 73:3, 61:2}
    snapnums_to_z_dict = {90:0.0, 73:0.5, 61:1.0}
    redshift = snapnums_to_z_dict[snapnum_hres]
    if isfid:
        cosmo_val_all = np.array([0.3175, 0.049, 0.6711, 0.9624, 0.834])
    else:
        cosmo_val_all = np.loadtxt(LH_cosmo_val_file)[isim]

    Om0 = cosmo_val_all[0]
    Ob0 = cosmo_val_all[1]
    h0 = cosmo_val_all[2]
    ns = cosmo_val_all[3]
    sigma8 = cosmo_val_all[4]
    params = {'flat': True, 'H0': h0*100, 'Om0': Om0, 'Ob0': Ob0, 'sigma8': sigma8, 'ns': ns}
    cosmo = cosmology.setCosmology('myCosmo', **params)

    snap_dir_base = '/mnt/home/fvillaescusa/ceph/Quijote/Halos/Rockstar/latin_hypercube_HR'
    verbose = False   #print information on progress
    snapdir = snap_dir_base + '/' + str(isim)  #folder hosting the catalogue
    rockstar = np.loadtxt(snapdir + '/out_' + str(snapnum_hres_dict[snapnum_hres]) + '_pid.list')
    with open(snapdir + '/out_' + str(snapnum_hres_dict[snapnum_hres]) + '_pid.list', 'r') as f:
        lines = f.readlines()
    header = lines[0].split()
    # get the properties of the halos
    pos_h_truth = rockstar[:,header.index('X'):header.index('Z')+1]
    index_M = header.index('M200c')                    
    mass_truth = rockstar[:,index_M]  #Halo masses in Msun/h
    lgMass_truth = np.log10(mass_truth).astype(np.float32)
    vel_h_truth = rockstar[:,header.index('VX'):header.index('VZ')+1]/vel_norm
    Rhalo = (1+redshift)*mass_so.M_to_R(mass_truth, redshift, '200c')   
    Rs = rockstar[:,header.index('Rs')]  
    conc_sim = Rhalo/Rs
    pid = rockstar[:,-1]
    prop_truth_all = np.stack((lgMass_truth, vel_h_truth[:,0], vel_h_truth[:,1], vel_h_truth[:,2], conc_sim)).T
    indsel = np.where((pid == -1) & (lgMass_truth > Mstar_cut))[0]
    prop_truth_all = prop_truth_all[indsel,:]
    pos_h_truth = pos_h_truth[indsel,:]


    dim_pos = pos_h_truth.shape[1]
    dim_prop = prop_truth_all.shape[1]


    Nhalos_truth = np.float32(np.zeros((grid_tot, grid_tot, grid_tot)))
    MASL.NGP(np.float32(pos_h_truth), Nhalos_truth, BoxSize)
    Nhalos_truth_rs = mat_reshape(Nhalos_truth, grid, grid_sbox)

    nMax_points = int(np.amax(Nhalos_truth_rs))

    dfhalo_ngp_wxyz_props = np.float32(np.zeros((grid_tot, grid_tot, grid_tot, nMax_points, dim_pos + dim_prop)))
    NGP_xyz_prop(np.float32(pos_h_truth), np.float32(prop_truth_all), dfhalo_ngp_wxyz_props, BoxSize)
    dfhalo_ngp_wxyz_props_rs = mat_reshape(dfhalo_ngp_wxyz_props, grid, grid_sbox)


    if add_space_token:
        max_sentence_length = 1 + Npoints_max_per_subvol*dim_tot + 1 + (Npoints_max_per_subvol - 1)
    else:
        max_sentence_length = 1 + Npoints_max_per_subvol*dim_tot + 1

    bins_digitize = np.linspace(-1e-3, 1, nvocab)
    Ntot_sel_final = 0
    sentences_all = np.zeros((grid, grid, grid, max_sentence_length), dtype=np.int16)
    for jx in range(grid):
        for jy in range(grid):
            for jz in range(grid):
                all_points_props_here = dfhalo_ngp_wxyz_props_rs[jx, jy, jz,...]
                Npoints_here = Nhalos_truth_rs[jx, jy, jz]
                indsel = np.where(Npoints_here > 0)
                Npoints_sel = Npoints_here[indsel]
                Npoints_sel_tot = int(np.sum(Npoints_sel))
                all_points_props_here_sel = all_points_props_here[indsel]
                word_array_all = []
                if len(Npoints_sel) > 0:    
                    for jc1 in range(len(Npoints_sel)):
                        Npoints_jc = (Npoints_sel[jc1]).astype(np.int16)
                        for jc2 in range(Npoints_jc):
                            all_points_props_here_sel_per_point = (all_points_props_here_sel[jc1, jc2, :])
                            position_token = np.array([indsel[0][jc1],indsel[1][jc1],indsel[2][jc1]], dtype=np.int16)
                            props_tokens = []
                            all_props = all_points_props_here_sel_per_point[3:]
                            for jp in range(len(all_props)):
                                prop_norm = np.clip((all_props[jp] - prop_min[jp]) / (prop_max[jp] - prop_min[jp]), 0.001, 0.999)
                                prop_token = np.digitize(prop_norm, bins_digitize).astype(np.int16)
                                if prop_token == 0:
                                    print(jp, prop_norm, prop_token)
                                props_tokens.append(prop_token)
                            props_token = np.array(props_tokens, dtype=np.int16)
                            all_tokens = np.concatenate((position_token, props_tokens))
                            word_array_all.append(all_tokens)
                    word_array_all = np.array(word_array_all, dtype=np.int16)
                    tosort_token_all = word_array_all[:, ind_token_to_sort]
                    sort_inds = np.flip(np.argsort(tosort_token_all))
                    word_array_all = word_array_all[sort_inds]
                    if Npoints_sel_tot > Npoints_max_per_subvol:
                        # print(isim_fid, ' LH-SIM HAS MORE POINTS (',Npoints_sel_tot,Npoints_max_per_subvol, ') THAN MAXIMUM IN THE', jx, jy, jz, ' THIS SUBVOLUME!!! max-sent-length: ',max_sentence_length)
                        word_array_all = word_array_all[:Npoints_max_per_subvol]    
                    Ntot_sel_final += len(word_array_all)
                    if add_space_token:
                        space_array = (np.array(np.zeros(word_array_all.shape[0]) + space_token, dtype=np.int16))[:,None]
                        word_array_all_concat = np.concatenate((word_array_all, space_array), axis=1)
                        sentence_here = np.concatenate(([start_token],(word_array_all_concat).flatten()[:-1], [end_token]))
                    else:
                        sentence_here = np.concatenate(([start_token],(word_array_all).flatten(), [end_token]))
                else:
                    sentence_here = np.array([start_token, end_token], dtype=np.int16)

                npad = max_sentence_length - len(sentence_here)
                if npad > 0:
                    # pad_mat = np.array(np.zeros((npad, dim_tot)) + pad_token, dtype=np.int16)
                    # sentence_pad = pad_mat.flatten()
                    sentence_pad = np.array(np.zeros(npad) + pad_token, dtype=np.int16)
                    sentence_here = np.concatenate((sentence_here, sentence_pad))
                    
                sentences_all[jx, jy, jz] = sentence_here

    story_full = sentences_all.reshape((grid**3, max_sentence_length))
    if int(np.sum(Nhalos_truth)) > Ntot_sel_final:
        print(isim, np.round(int(np.sum(Nhalos_truth))/Ntot_sel_final, 3))

    saved = {
            'story_full': story_full.astype(np.int16)[rand_sel, ...],
            'max_sentence_length': max_sentence_length,
            'grid': grid,
            'grid_sbox': grid_sbox,
            'Npoints_max_per_subvol': Npoints_max_per_subvol,
            'BoxSize': BoxSize,
            'prop_min': prop_min,
            'prop_max': prop_max,
            'nvocab':nvocab,
            'start_token': start_token,
            'pad_token': pad_token,
            'end_token': end_token,
            'space_token': space_token,
            'bins_digitize':bins_digitize,
            'rand_sel': rand_sel
            }
    pk.dump(saved, open(savefname_gals, 'wb'))
    gc.collect()
    return



import multiprocessing as mp
if __name__ == '__main__':
    # n_sims = 1100
    # n_sims_offset = 0
    # n_sims = 1000
    n_sims_offset = int(sys.argv[-1])
    n_sims = int(sys.argv[-2])
    # n_cores = mp.cpu_count()
    n_cores = 2
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
