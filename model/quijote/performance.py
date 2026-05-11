import numpy as np
import Pk_library as PKL
import MAS_library as MASL
import redshift_space_library as RSL
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def hubble(redshift, Omega_m):
    # Planck 2018 cosmological parameters
    H0 = 100  # Hubble constant at z=0 in km/s/(Mpc/h)
    Omega_Lambda = 1.0 - Omega_m

    Hz = H0 * np.sqrt(Omega_m * (1 + redshift)**3 + Omega_Lambda)
    return Hz  # in km/s/(Mpc/h)


logmass_bins = np.linspace(12.7,15,20)
mass_bins = 10**logmass_bins
BoxSize = 1000.0  # Mpc/h
volum = BoxSize**3  # (Mpc/h)^3
def hmf(data):
    counts, _ = np.histogram(data, bins=mass_bins)
    dn_dlogm = counts / volum / np.diff(logmass_bins)
    return dn_dlogm

def hist(data,bins):
    counts, _ = np.histogram(data, bins=bins)
    return counts

def power_spectrum(pos, weight=None):
    pos = np.float32(pos)
    Ngrid = 512
    MAS = 'NGP'
    delta = np.zeros((Ngrid,Ngrid,Ngrid),dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, verbose=False, W=weight)
    delta /= np.mean(delta, dtype=np.float32);  delta -= 1.0
    Pk = PKL.Pk(delta, BoxSize, axis=2, MAS='None', threads=32, verbose=False)
    k = Pk.k3D
    Pk0 = Pk.Pk[:,0]
    return k, Pk0

grid = 64
nvocab = 131
pos_vocab = 40
add_space_token = False
dim_tot = 8
start_token = nvocab + 1
space_token = nvocab + 2
pad_token = nvocab + 3
end_token = nvocab + 4
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
                    # try:
                    Nhalos_here = ((ind_end_token - ind_start_token - 1) / ntokens_per_halo)
                    # except:
                        # print(ind_end_token, ind_start_token, ntokens_per_halo)
                    if (int(Nhalos_here) - Nhalos_here) != 0:
                        #print(Nhalos_here, 'Nhalos_here is not an integer')
                        Nhalos_here = int(Nhalos_here)
                    else:
                        if Nhalos_here > 0:
                            for jh in range(int(Nhalos_here)):
                                try:
                                    prop_all = np.zeros(5)        
                                    for jp in range(dim_prop):
                                        # noise_val = np.random.uniform(-delta_bin/2, delta_bin/2)
                                        bin_val_jp = sentence_here[ind_start_token + jh*ntokens_per_halo + 4 + jp]
                                        prop_all[jp] = (bins_digitize[jp, bin_val_jp] +np.random.uniform(-0.5,0.5) * bins_step[jp]).clip(min=bins_digitize[jp,0], max=bins_digitize[jp,-1])

                                    coord_x = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 1]] + (BoxSize/grid)*jx + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                    coord_y = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 2]] + (BoxSize/grid)*jy + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                    coord_z = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 3]] + (BoxSize/grid)*jz + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                    pos_infer_all.append([coord_x, coord_y, coord_z])
                                            # print(sentence_here)
                                    prop_infer_all.append(prop_all)

                                except Exception as e:
                                    
                                    pass
                else:           
                    #print('End token not found')
                    pass
    #pos_infer_all = np.concatenate(pos_infer_all, axis=0)
    #prop_infer_all = np.concatenate(prop_infer_all, axis=0)
    return np.array(pos_infer_all), np.array(prop_infer_all)


def get_prop_pos_xfirst(X_val):
    pos_infer_all = []
    prop_infer_all = []
    
    for jx in range(grid):
        for jy in range(grid):
            for jz in range(grid):
                sentence_here = X_val[jx, jy, jz]
                ind_start_token = np.where(sentence_here == start_token)[0][0]
                
                if end_token in sentence_here:
                    ind_end_token = np.where(sentence_here == end_token)[0][0]
                    # try:
                    Nhalos_here = ((ind_end_token - ind_start_token -2) / dim_tot)
                    # except:
                        # print(ind_end_token, ind_start_token, ntokens_per_halo)
                    if (int(Nhalos_here) - Nhalos_here) != 0 or Nhalos_here != sentence_here[1]:
                        print(f'Nhalos_here = {Nhalos_here}, but sentence indicates {sentence_here[1]}')
                        
                    else:
                        Nhalos_here = int(Nhalos_here)
                        if Nhalos_here > 0:
                            try:
                                x_tokens = sentence_here[2:2+Nhalos_here]
                                coord_xs = (xarray[x_tokens] + (BoxSize/grid)*jx + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                y_tokens = sentence_here[2+Nhalos_here:2+2*Nhalos_here]
                                coord_ys = (xarray[y_tokens] + (BoxSize/grid)*jy + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                z_tokens = sentence_here[2+2*Nhalos_here:2+3*Nhalos_here]
                                coord_zs = (xarray[z_tokens] + (BoxSize/grid)*jz + np.random.uniform(-0.5,0.5)*dx) % BoxSize
                                props = np.zeros((dim_prop, Nhalos_here))
                                for jp in range(dim_prop):
                                    tokens_jp = sentence_here[2+(jp+3)*Nhalos_here:2+(jp+4)*Nhalos_here]
                                    props[jp] = (bins_digitize[jp, tokens_jp] +np.random.uniform(-0.5,0.5,size=Nhalos_here) * bins_step[jp]).clip(min=bins_digitize[jp,0], max=bins_digitize[jp,-1])
                                
                                pos_infer_all.extend(np.stack((coord_xs, coord_ys, coord_zs), axis=-1))
                                prop_infer_all.extend(props.T)
                            except Exception as e:
                                pass

    return np.array(pos_infer_all), np.array(prop_infer_all)

def get_ratios(simid, gen_data, xfirst=False):
    all_param = np.loadtxt('/work/nvme/bdne/yzhang116/quijote_halos/quijote_params.txt')
    Om = all_param[simid][0]
    mcrit = 10**12.7
    if xfirst == True:
        gen_data = np.delete(gen_data, [1,2,3,4,5], axis=1)
        gen_data = gen_data.reshape((grid,grid,grid,-1))
        pos, prop = get_prop_pos_xfirst(gen_data)
    else:
        gen_data = np.delete(gen_data, [1,2,3,4,5,6], axis=1)
        gen_data = gen_data.reshape((grid,grid,grid,-1))
        pos, prop = get_prop_pos(gen_data)
    vel = prop[:,1:4]
    data = np.loadtxt('/work/hdd/bdne/yzhang116/quijote/halo_catalogs/halo_LH_%d.dat'%simid)
    pos_true = data[:, 1:4]
    mass     = data[:, 0]
    vel_true = data[:, 4:7]
    index = np.where(mass > mcrit)[0]
    pos_true = pos_true[index]
    mass = mass[index]
    num_true = mass.shape[0]
    num_infer = prop.shape[0]
    pos = np.float32(pos)
    pos_true = np.float32(pos_true)
    print(f'Sim {simid}: true halos = {num_true}, inferred halos = {num_infer}')
    hmf1 = hmf(np.power(10, prop[:,0]))
    hmf2 = hmf(mass)
    hmf_ratio = (hmf1/hmf2-1)*100
    RSL.pos_redshift_space(pos,np.float32(vel), BoxSize, hubble(0.5,Om), 0.5, axis=2)
    RSL.pos_redshift_space(pos_true, np.float32(vel_true), BoxSize, hubble(0.5,Om), 0.5, axis=2)
    k, Pk0 = power_spectrum(pos)
    k1, Pk1 = power_spectrum(pos_true)
    #Pk0 -= (BoxSize**3/num_infer)
    #Pk1 -= (BoxSize**3/num_true)
    pk_ratio = (Pk0/Pk1-1)*100
    return np.array([logmass_bins[:-1],hmf_ratio]),np.array([k,pk_ratio]), num_infer


def plot_log(log_ps, log_hmf,hnum,filename):
    norm = colors.Normalize(vmin=1e5, vmax=1e6)
    cmap = cm.get_cmap('viridis')
    fig, axs = plt.subplots(1,2, figsize=(11,5))
    num = log_ps.shape[0]
    for i in range(num):
        axs[0].plot(log_hmf[i,0], log_hmf[i,1], alpha=0.5, c=cmap(norm(hnum[i])))
        axs[1].plot(log_ps[i,0], log_ps[i,1], alpha=0.5, c=cmap(norm(hnum[i])))

    axs[0].axhline(1.0, color='k', linestyle='--')
    axs[0].set_xlabel('log10(Mass / Msun/h)')
    axs[0].set_ylabel('HMF Percentage difference (%)')
    axs[0].set_ylim(-50,50)
    axs[0].grid()

    axs[1].axhline(1.0, color='k', linestyle='--')
    axs[1].set_xlabel('k [h/Mpc]')
    axs[1].set_ylabel('P(k) Percentage difference (%)')
    axs[1].set_xscale('log')
    axs[1].set_ylim(-50,50)
    axs[1].grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

