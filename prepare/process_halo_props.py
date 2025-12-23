import numpy as np
import sys,os
import MAS_library as MASL
import pickle as pk
from ngp_funcs import NGP_xyz_prop 
import skimage.measure as skmeasure
import ast
from colossus.halo import mass_so
from colossus.cosmology import cosmology 
import gc
import mpi4py.MPI as MPI

###### MPI DEFINITIONS ######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

nstart = 0
nend   = 1000

numbers = np.arange(nstart, nend, dtype=int)
numbers = numbers[numbers % nprocs == myrank]



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


param_file = '/work/hdd/bdne/yzhang116/CAMEL_SAM_params.txt'
Oms, sigma8s = np.loadtxt(param_file, usecols=(1,2), dtype=np.float32,unpack=True)

grid = 16
nrand_sel_box = grid**3
add_space_token = False
# nrand_sel_box = 512 
BoxSize = 100.
nvocab = 64
pos_nvocab = 64


# Mstar_cut = 12.5
grid_sbox = pos_nvocab
grid_tot = grid_sbox * grid
#vel_norm = 2000.
#prop_min = np.array([11.5, -0.6, -0.6, -0.6, 1.0])
#prop_max = np.array([14.5, 0.6, 0.6, 0.6, 20.0])

redshift = 0.5
Npoints_max_per_subvol = 36
#sort by Mstar token:
ind_token_to_sort = 3
dim_pos = 3
dim_prop = 5
dim_tot = dim_pos + dim_prop
start_token = nvocab + 1
space_token = nvocab + 2
pad_token = nvocab + 3
end_token = nvocab + 4
#bins_digitize = np.linspace(-1e-3, 1, nvocab)
bins_digitize = np.zeros((dim_prop, nvocab))
v_left = np.linspace(-1200, -100, 22, endpoint=False)
v_mid = np.linspace(-100,  100, 19, endpoint=False) 
v_right = np.linspace(100, 1200, 23, endpoint=True) 
bins_digitize[0] = np.linspace(11.5, 14.5, nvocab)  # Mass
bins_digitize[4] = np.linspace(1.0, 30.0, nvocab)  # concentration
for i in range(1,4):
    bins_digitize[i] = np.concatenate((v_left, v_mid, v_right))
Om_min = 0.1
Om_max = 0.5
sigma8_min = 0.6
sigma8_max = 1.0
costoken_num = 2
rand_sel = np.arange(grid**3)
if add_space_token:
    max_sentence_length = 1 + Npoints_max_per_subvol*dim_tot + 1 + (Npoints_max_per_subvol - 1) + costoken_num
else:
    max_sentence_length = 1 + Npoints_max_per_subvol*dim_tot + 1 + costoken_num

saved = {
    'max_sentence_length': max_sentence_length,
    'grid': grid,
    'grid_sbox': grid_sbox,
    'Npoints_max_per_subvol': Npoints_max_per_subvol,
    'BoxSize': BoxSize,
    #'prop_min': prop_min,
    #'prop_max': prop_max,
    'nvocab':nvocab,
    'pos_nvocab': pos_nvocab,
    'start_token': start_token,
    'pad_token': pad_token,
    'end_token': end_token,
    'space_token': space_token,
    'bins_digitize':bins_digitize,
    'rand_sel': rand_sel
    }
pk.dump(saved, open("/work/hdd/bdne/yzhang116/halo_sentences_11.5/sentence_params_36.pkl", 'wb'))
print("Sentence parameters saved.")


def process_LH_sim(isim):
    from colossus.cosmology import cosmology
    from colossus.halo import mass_so  

    input_file = '/work/hdd/bdne/yzhang116/halo_catalogs_11.5/halo_LH_%d.dat'%isim
    #input_file = '/work/hdd/bdne/yzhang116/CV/halo_catalogs_CV/halo_CV_%d.dat'%isim

    params = {'flat': True, 'H0': 67.11, 'Om0': Oms[isim], 'Ob0': 0.049, 'sigma8': sigma8s[isim], 'ns': 0.9624}
    #params = {'flat': True, 'H0': 67.11, 'Om0': 0.3, 'Ob0': 0.049, 'sigma8': 0.8, 'ns': 0.9624}
    cosmo = cosmology.setCosmology('myCosmo', persistence='',**params)
    
    sdir = '/work/hdd/bdne/yzhang116/halo_sentences_11.5'
    savefname_gals = f'{sdir}/halo_sentence36_LH_{isim}.npy'
    #sdir = '/work/hdd/bdne/yzhang116/CV/halo_sentences'
    #savefname_gals = f'{sdir}/halo_sentence_CV_{isim}.npy'

    Om_token = np.round((Oms[isim] - Om_min) / (Om_max - Om_min) * nvocab).astype(np.int16)
    Om_token = np.clip(Om_token, 0, nvocab)
    sigma8_token = np.round((sigma8s[isim] - sigma8_min)/(sigma8_max - sigma8_min) * nvocab).astype(np.int16)
    sigma8_token = np.clip(sigma8_token, 0, nvocab)

    # get the properties of the halos
    data = np.loadtxt(input_file, dtype=np.float32)
    pos_h_truth = data[:,1:4]
    mass_truth = data[:,0]
    lgMass_truth = np.log10(mass_truth).astype(np.float32)
    vel_h_truth = data[:,4:7] # / vel_norm
    Rhalo = (1+redshift)*mass_so.M_to_R(mass_truth, redshift, '200c')
    conc_sim = Rhalo / data[:,7] # R_200c / Rs
    prop_truth_all = np.stack((lgMass_truth, vel_h_truth[:,0], vel_h_truth[:,1], vel_h_truth[:,2], conc_sim)).T
    dim_pos = pos_h_truth.shape[1]
    dim_prop = prop_truth_all.shape[1]

    #indsel = np.where(lgMass_truth > Mstar_cut)[0]
    #prop_truth_all = prop_truth_all[indsel,:]
    #pos_h_truth = pos_h_truth[indsel,:]

    Nhalos_truth = np.float32(np.zeros((grid_tot, grid_tot, grid_tot)))
    MASL.NGP(np.float32(pos_h_truth), Nhalos_truth, BoxSize)
    Nhalos_truth_rs = mat_reshape(Nhalos_truth, grid, grid_sbox)

    nMax_points = int(np.amax(Nhalos_truth_rs))

    dfhalo_ngp_wxyz_props = np.float32(np.zeros((grid_tot, grid_tot, grid_tot, nMax_points, dim_pos + dim_prop)))
    NGP_xyz_prop(np.float32(pos_h_truth), np.float32(prop_truth_all), dfhalo_ngp_wxyz_props, BoxSize)
    dfhalo_ngp_wxyz_props_rs = mat_reshape(dfhalo_ngp_wxyz_props, grid, grid_sbox)

    del data, pos_h_truth, mass_truth, lgMass_truth, vel_h_truth, Rhalo, conc_sim, prop_truth_all
    gc.collect()



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
                                prop_token = np.digitize(all_props[jp], bins_digitize[jp]).astype(np.int16)
                                #prop_norm = np.clip((all_props[jp] - prop_min[jp]) / (prop_max[jp] - prop_min[jp]), 0.001, 0.999)
                                #prop_token = np.digitize(prop_norm, bins_digitize).astype(np.int16)
                                #if prop_token == 0:
                                #    print(jp, prop_norm, prop_token)
                                props_tokens.append(prop_token)
                            props_token = np.array(props_tokens, dtype=np.int16)
                            all_tokens = np.concatenate((position_token, props_tokens))
                            word_array_all.append(all_tokens)
                    word_array_all = np.array(word_array_all, dtype=np.int16)
                    tosort_token_all = word_array_all[:, ind_token_to_sort]
                    sort_inds = np.flip(np.argsort(tosort_token_all))
                    word_array_all = word_array_all[sort_inds]
                    if Npoints_sel_tot > Npoints_max_per_subvol:
                        print(isim, ' LH-SIM HAS MORE POINTS (',Npoints_sel_tot,Npoints_max_per_subvol, ') THAN MAXIMUM IN THE', jx, jy, jz, ' THIS SUBVOLUME!!! max-sent-length: ',max_sentence_length)
                        word_array_all = word_array_all[:Npoints_max_per_subvol]    
                    Ntot_sel_final += len(word_array_all)
                    if add_space_token:
                        space_array = (np.array(np.zeros(word_array_all.shape[0]) + space_token, dtype=np.int16))[:,None]
                        word_array_all_concat = np.concatenate((word_array_all, space_array), axis=1)
                        sentence_here = np.concatenate(([start_token],[Om_token,sigma8_token],(word_array_all_concat).flatten()[:-1], [end_token]))
                    else:
                        sentence_here = np.concatenate(([start_token],[Om_token,sigma8_token],(word_array_all).flatten(), [end_token]))
                else:
                    sentence_here = np.array([start_token,Om_token,sigma8_token, end_token], dtype=np.int16)

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
    np.save(savefname_gals, story_full.astype(np.int16)[rand_sel, ...])
    
    del dfhalo_ngp_wxyz_props, dfhalo_ngp_wxyz_props_rs, Nhalos_truth, Nhalos_truth_rs,story_full,sentences_all
    gc.collect()
    return

for i in numbers:
    process_LH_sim(i)
