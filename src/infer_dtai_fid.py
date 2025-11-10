import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
import numpy as np
import torch
import yaml
import pickle as pk
import numpy as np
import sys,os
import readgadget
import MAS_library as MASL
import pickle as pk
import readfof
import matplotlib
import readfof
import sys, os
import numpy as np
import pickle as pk 
# from nbodykit.lab import *
import h5py as h5
import numpy as np
import Pk_library as PKL
import MAS_library as MASL
import yaml
from train_dtai import *
from tqdm import tqdm
import gc
        
dev = torch.device("cuda")
# checkpoint = torch.load('/projects/bdne/spandey3/GOTHAM/model_checkpoints/camels_photo/model_encdec_ddp_PM_nvocab_64_nembed_64_nrandsubsel_32_iter_900.pt')
# checkpoint = torch.load('/projects/bdne/spandey3/GOTHAM/model_checkpoints/camels_photo/model_encdec_ddp_PM_nvocab_64_nembed_64_nrandsubsel_32.pt')

subsel_type = 'all'
# subsel_type = 'no_env'
# subsel_type = 'no_vel'
# subsel_type = 'no_highz_nsnap_3'
# Mstar_cut = 8.5
Mstar_cut =12.7
add_space_token = False
# grid_sbox = 32
grid_sbox = 8

isfid=True

# checkpoint = torch.load(f'/projects/bdne/spandey3/GOTHAM/model_checkpoints/camels_photo_velx/all_run2.pt')
checkpoint = torch.load('/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/quijote_halos/FINAL4_fidfinetune_TEST_model_hres_encdec_ddp_grid_8_nvocab_64_nembed_256_nhead_8_nrandsubsel_8192_subselDMOfields_all_Mstarcut_12.7_spacetoken_False_maxiter_1500_lr_0.0005.pt')
# subsel_type = 'no_highz'
# checkpoint = torch.load('/projects/bdne/spandey3/GOTHAM/model_checkpoints/camels_photo_velx/model_hres_encdec_ddp_PM_nvocab_64_nembed_128_nhead_8_nrandsubsel_64_subselDMOfields_no_highz.pt')

HaloConfig = checkpoint['config']
print(checkpoint['best_val_loss'])
learning_rate = 5e-4
model = HaloDecoderModel(HaloConfig).to(dev)
# if rank == 0: print(f"Init model and loaded to GPU", flush=True)            
# model.to(device).bfloat16()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# checkpoint = torch.load('/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec_ddp_PM_isim_012_nvocab_96_nembed_64_CORRMASK_NOFLASH_FINAL.pt')
# checkpoint = torch.load('/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec_ddp_PM_isim_012_nvocab_64_nembed_64_Mmin_13p0.pt')
# checkpoint = torch.load('/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec_ddp_PM_isim_012_nvocab_64_nembed_96_Mmin_13p0.pt')
# checkpoint = torch.load('/mnt/home/spandey/ceph/CHARFORMER/model_checkpoints/model_encdec_ddp_PM_isim_012_nvocab_64_nembed_64_Mmin_13p5.pt')
# checkpoint = torch.load('/mnt/home/spandey/ceph/GOTHAM/model_checkpoints/camels/model_encdec_ddp_PM_nvocab_64_nembed_64_nrandsubsel_8_iter_1000.pt')
model.load_state_dict(checkpoint['model'])
model.eval()
# print()
model = model.bfloat16()

params_all = np.loadtxt('/projects/bdne/spandey3/halo_gotham/GOTHAM/prep_data/latin_hypercube_params.txt')


# from tqdm import tqdm
TEMPERATURE = 1.0
nvocab = 64
grid = 32
# add_space_token = False
BoxSize = 1000.
xall = (np.linspace(0, BoxSize/grid, nvocab + 1))
xarray = 0.5 * (xall[1:] + xall[:-1])
yarray = np.copy(xarray)
zarray = np.copy(xarray)

from tqdm import tqdm
saved_all = {}
# isim_fid_array = np.arange(936, 946)
# isim_fid_array = np.arange(975, 990)
# isim_fid_array = np.arange(990, 996)
# isim_fid_array = np.arange(113, 114)
isim_fid_array = np.arange(16, 17)
# isim_fid_array = np.arange(93, 94)
for isim_fid in (isim_fid_array):


    ldir = '/work/hdd/bdne/spandey3/quijote_data/halo_gotham_data/LH'
    if isfid:
        savefname_dmo_fields = f'{ldir}/DMO_fields_ng32/DMO_fields_grid_{grid_sbox}_fidcosmo_isim_{isim_fid}_nrandsubsel_32768_MAS_CIC_nsnaps_2.pkl'
    else:
        savefname_dmo_fields = f'{ldir}/DMO_fields_ng32/DMO_fields_grid_{grid_sbox}_isim_{isim_fid}_nrandsubsel_32768_MAS_CIC_nsnaps_2.pkl'
    # df = pk.load(open(f'{sdir}/subhalo_density3Dgrid_32_isim_{isim_fid}_nrandsubsel_512_nvocab64_wSDSS_photometry_vel_hres.pkl','rb'))    
    # delta_box_all_squeezed_0 = df['delta_box_all_squeezed']
    df = pk.load(open(savefname_dmo_fields,'rb'))
    dmo_fields_all = torch.moveaxis(torch.tensor(df['dmo_fields_all']).to(torch.float16), -1, 1)

    # savefname_gals = f'{ldir}/gal_props/galaxy_props_snap_90_grid_{nvocab}_isim_{isim_fid}_nrandsubsel_512_nvocab{nvocab}_wSDSS_photometry_velx.pkl'
    if isfid:
        savefname_gals = f'{ldir}/halo_props_ng32/halo_props_snap_73_grid_{nvocab}_fidcosmo_isim_{isim_fid}_nrandsubsel_32768_nvocab{nvocab}_spacetoken_{add_space_token}_xMvc_{Mstar_cut}.pkl'    
    else:
        savefname_gals = f'{ldir}/halo_props_ng32/halo_props_snap_73_grid_{nvocab}_isim_{isim_fid}_nrandsubsel_32768_nvocab{nvocab}_spacetoken_{add_space_token}_xMvc_{Mstar_cut}.pkl'    
    df = pk.load(open(savefname_gals,'rb'))
    dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_0 = df['story_full']

    if isfid:
        param_0 = np.array([0.3175, 0.049, 0.6711, 0.9624, 0.834])[None, :]
    else:
        param_0 = params_all[isim_fid][None, :]
    param_0 = np.repeat(param_0, len(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_0), axis=0)

    # grid = 8
    # shift_x_all, shift_y_all, shift_z_all = np.zeros((grid, grid, grid)), np.zeros((grid, grid, grid)), np.zeros((grid, grid, grid))
    # for jx in range(grid):
    #     for jy in range(grid):
    #         for jz in range(grid):
    #             shift_x_all[jx, jy, jz] = jx
    #             shift_y_all[jx, jy, jz] = jy
    #             shift_z_all[jx, jy, jz] = jz

    # shift_x_all_squeezed, shift_y_all_squeezed, shift_z_all_squeezed = shift_x_all.flatten(), shift_y_all.flatten(), shift_z_all.flatten()
    # shift_all_squeezed = np.vstack([shift_x_all_squeezed, shift_y_all_squeezed, shift_z_all_squeezed]).T


    n1 = 32**3
    test_data_halos = dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_0[:n1]

    test_data_dm = dmo_fields_all[:n1]
    params_test = param_0[:n1]

    x = torch.tensor(test_data_halos[:, :-1])
    y = torch.tensor(test_data_halos[:, 1:])
    dm = torch.tensor(test_data_dm)
    mask_test_orig = x != 1
    mask_test = torch.logical_not(mask_test_orig)
    masked_logits = torch.zeros(mask_test.shape)
    mask_test_final = masked_logits.masked_fill(mask_test, float('-inf'))
    mask_test = mask_test_final[:,None,:]
    x, y = torch.tensor(x), torch.tensor(y)
    x_test = x.long()
    y_test = y.long()
    dm_test = dm.bfloat16()
    mask_test = torch.tensor(mask_test).bfloat16()
    params_test = torch.tensor(params_test).bfloat16()

    x_test_gpu = x_test.to(dev)
    y_test_gpu = y_test.to(dev)
    dm_test_gpu = dm_test.to(dev)
    mask_test_gpu = mask_test.to(dev)
    params_test_gpu = params_test.to(dev)

    # if subsel_type == 'no_highz':
    #     indices = torch.arange(6)
    # if subsel_type == 'no_highz_nsnap_2':
    #     indices = torch.arange(12)        
    # if subsel_type == 'no_highz_nsnap_3':
    #     indices = torch.arange(18)                
    # elif subsel_type == 'no_highz_no_vel':
    #     indices = torch.arange(3)
    # elif subsel_type == 'no_highz_no_env':
    #     indices = torch.from_numpy(np.array([0,3,4,5]))
    # elif subsel_type == 'no_vel':
    #     indices = torch.cat([torch.arange(i, i + 3) for i in range(0, 30, 6)])
    # elif subsel_type == 'no_env':        
    #     indices1 = torch.cat([torch.arange(i+3, i + 6) for i in range(0, 30, 6)])
    #     indices2 = torch.cat([torch.arange(i, i + 1) for i in range(0, 30, 6)])
    #     indices, _ = torch.sort(torch.cat([indices1, indices2]))
    # else:
    indices = torch.arange(dm_test_gpu.shape[1])


    dm_test_gpu = dm_test_gpu[:,indices,...]
    

    def get_batch(split, ji=0, batch_size=None):
        if split == 'test':
            x = x_test_gpu
            y = y_test_gpu
            mask = mask_test_gpu
            dm = dm_test_gpu       
            params = params_test_gpu 

        if batch_size is not None:
            x = x[batch_size*(ji):batch_size*(ji+1)].to(dev, non_blocking=True)
            y = y[batch_size*(ji):batch_size*(ji+1)].to(dev, non_blocking=True)
            mask = mask[batch_size*(ji):batch_size*(ji+1)].to(dev, non_blocking=True)
            dm = dm[batch_size*(ji):batch_size*(ji+1)].to(dev, non_blocking=True)
            params = params[batch_size*(ji):batch_size*(ji+1)].to(dev, non_blocking=True)

        return x, y, mask, dm, params

    # batch_size = 8**3
    batch_size = 32**3    
    X_val, Y_val, MASK_val, DM_val, PARAMS = get_batch('test', 0, batch_size)
    # DM_val = torch.moveaxis(DM_val, -1, 1)

    max_new_tokens = df['max_sentence_length']
    nvocab_tot = df['end_token'] + 1

    bins_digitize = np.linspace(-1e-3, 1, nvocab)
    # bins_digitize.insert(0, -1)
    # bins_digitize = np.insert(bins_digitize, 0, -1)


    start_token = df['start_token']
    pad_token = df['pad_token']
    end_token =  df['end_token']
    space_token = df['space_token']


    from tqdm import tqdm

    fac = 1
    nvox_samp = batch_size
    nbatches = fac
    nvox_per_batch = nvox_samp // nbatches

    # idx_all = []
    # idx_all = torch.ones((nvox_samp, max_new_tokens), dtype=torch.long, device=dev)
    idx_all = np.ones((nvox_samp, max_new_tokens))
    
    for jb in tqdm(range(nbatches)):

        idx_inp = torch.zeros((nvox_per_batch, 1), dtype=torch.long, device=dev)

        DM_val_jb = DM_val[jb*nvox_per_batch:(jb+1)*nvox_per_batch,...]

        param_jb = PARAMS[jb*nvox_per_batch:(jb+1)*nvox_per_batch,...]

        new_samples_jb = np.ones((nvox_per_batch, max_new_tokens))

        ind_jb = np.arange(nvox_per_batch)

        new_samples_jb[:, 0] = idx_inp[:,0].cpu().detach().numpy() + start_token
        # idx = idx_inp
        for jt in range(1, max_new_tokens):
            if len(ind_jb) > 0:
                # crop idx to the last block_size tokens
                idx_cond = torch.tensor(new_samples_jb[ind_jb, :jt], dtype=torch.long, device=dev)
                # get the predictions
                with torch.no_grad():
                    logits, loss = model(idx_cond, DM_val_jb[ind_jb,...], params=param_jb[ind_jb,...])
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits/TEMPERATURE, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1).cpu().detach().numpy() # (B, 1)

                # append sampled index to the running sequence
                # if idx_next == end_token:
                    # break
                new_samples_jb[ind_jb, jt] = idx_next[:,0]
                # ind_to_del = np.where(idx_next[:,0] == end_token)[0]
                ind_to_del = np.where((idx_next[:,0] == end_token) | (idx_next[:,0] == pad_token))[0]
                # for jv in range(len(ind_jb)):
                    # if idx_next[jv, 0] == end_token:
                if len(ind_to_del) > 0:
                    ind_jb = np.delete(ind_jb, ind_to_del)

                torch.cuda.empty_cache()
                gc.collect()
                


            # print(new_samples_jb[0,:])
            # idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        # print(new_samples_jb)
        idx_all[jb*nvox_per_batch:(jb+1)*nvox_per_batch, :] = new_samples_jb
        # idx_all.append(idx)
    # print(idx_all)
    dm_mean_allvox = np.mean(np.array(DM_val[:,1,...].to(torch.float16).detach().cpu()), axis=(1,2,3))

    prop_min = df['prop_min']
    prop_max = df['prop_max']
    
    dim_pos = 3
    dim_prop = len(prop_min)   
    dim_tot = dim_pos + dim_prop
    

    # BoxSize = 1000.
    # grid = 8
    xmin = BoxSize/grid/2

    MAS     = 'NGP'  #mass-assigment scheme
    # grid_Pk = 16
    # threads = 10
    
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
                        ind_end_token = np.where(sentence_here == end_token)[0]
                        try:
                            Nhalos_here = ((ind_end_token - ind_start_token - 1) / ntokens_per_halo)[0]
                        except:
                            print(ind_end_token, ind_start_token, ntokens_per_halo)
                        if (int(Nhalos_here) - Nhalos_here) != 0:
                            Nhalos_here = ((ind_end_token - ind_start_token ) / ntokens_per_halo)[0]
                            # print(Nhalos_here, Nhalos_here2, 'Nhalos_here is not an integer')
                            # pass
                        if (int(Nhalos_here) - Nhalos_here) != 0:
                            print(Nhalos_here, 'Nhalos_here is not an integer')
                        else:
                            if Nhalos_here > 0:
                                for jh in range(int(Nhalos_here)):
                                    try:
                                        coord_x = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 1]] + (BoxSize/grid)*jx) % BoxSize
                                        coord_y = (yarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 2]] + (BoxSize/grid)*jy) % BoxSize
                                        coord_z = (zarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 3]] + (BoxSize/grid)*jz) % BoxSize
                                        pos_infer_all.append([coord_x, coord_y, coord_z])
                                        prop_all = np.zeros(dim_prop)        
                                        for jp in range(dim_prop):
                                            # noise_val = np.random.uniform(-delta_bin/2, delta_bin/2)
                                            noise_val = 0
                                            bin_val_jp = sentence_here[ind_start_token + jh*ntokens_per_halo + 4 + jp]
                                            try:
                                                prop_all[jp] = prop_min[jp] + (prop_max[jp] - prop_min[jp]) * (bins_digitize[bin_val_jp] + noise_val)
                                            except:
                                                pass
                                                # print(sentence_here)
                                        prop_infer_all.append(prop_all)
     
                                    except:
                                        pass
                                        # print(sentence_here[ind_start_token], ntokens_per_halo)
                    else:           
                        # print('End token not found')
                        pass
        
        pos_infer_all = np.array(pos_infer_all)
        prop_infer_all = np.array(prop_infer_all)
        return pos_infer_all, prop_infer_all

    # print(idx_all, X_val)
    idx_all_rs = np.reshape(np.array(idx_all), (grid, grid, grid, idx_all.shape[-1])).astype(int)
    # X_val_rs = np.reshape(X_val.cpu().numpy(), (grid, grid, grid, X_val.shape[-1])).astype(int)
    X_val_rs = np.reshape(dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_0, (grid, grid, grid, dfhalo_ngp_xyzM_tokenized_padded_ended_squeezed_0.shape[-1])).astype(int)
    print('getting inferred positions and properties')
    pos_infer_all, prop_infer_all = get_prop_pos(idx_all_rs)

    print('getting truth positions and properties')
    pos_truth_all, prop_truth_all = get_prop_pos(X_val_rs)

    # snapnum = 90
    # ldir = '/work/hdd/bdne/spandey3/camels_tng/hydro'
    # group_catalog = f'{ldir}/LH/LH_{isim_fid}/groups_0{snapnum}.hdf5'
    # photo_catalog = f'{ldir}/Photometry/IllustrisTNG/L25n256/LH/IllustrisTNG_LH_{isim_fid}_photometry.hdf5'
    # # open the catalogue
    # with h5.File(photo_catalog, "r") as hf:
    #     subhalo_index = np.array(hf[f"snap_0{snapnum}/SubhaloIndex"][:], dtype=int)
    #     g_band = np.log10(hf[f"snap_0{snapnum}/BC03/photometry/luminosity/attenuated/SLOAN/SDSS.g"][:])
    #     r_band = np.log10(hf[f"snap_0{snapnum}/BC03/photometry/luminosity/attenuated/SLOAN/SDSS.r"][:])
    #     i_band = np.log10(hf[f"snap_0{snapnum}/BC03/photometry/luminosity/attenuated/SLOAN/SDSS.i"][:])
    
    # # Read the stellar masses of the subhalos/galaxies
    # with h5.File(group_catalog, "r") as hf:
    #     M_star = np.log10(hf['Subhalo/SubhaloMassType'][:,4]*1e10 + 0.01) # Stellar masses in Msun/h
    #     pos = hf['Subhalo/SubhaloPos'][:]/1000.
    #     vel = hf['Subhalo/SubhaloVel'][:]
    
    # M_star = M_star[subhalo_index]
    # pos_truth_orig = pos[subhalo_index]
    # vel_truth_orig = vel[subhalo_index][:,0]
    # indsel = np.where(M_star > Mstar_cut)[0]
    # print(pos_truth_orig[indsel,:].shape, pos_truth_all.shape, pos_infer_all.shape)
    
    saved_all[isim_fid] = {'pos_truth':pos_truth_all, 'prop_truth':prop_truth_all, 'pos_infer':pos_infer_all, 'prop_infer':prop_infer_all}


import pickle as pk
with open(f'/projects/bdne/spandey3/halo_gotham/GOTHAM/infer_data/FINAL4_fidfinetune_TEST_infer_halos_ng32_grid_{grid_sbox}_isim_fid_{isim_fid_array[0]}_{isim_fid_array[-1]}_nrandsubsel_32768_Mstarcut_{Mstar_cut}_spacetoken_{add_space_token}.pkl', 'wb') as f:
    pk.dump(saved_all, f)