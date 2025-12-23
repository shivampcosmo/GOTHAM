import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import Pk_library as PKL
import MAS_library as MASL
import redshift_space_library as RSL
import os
import glob
import random

def hubble(redshift, Omega_m):
    # Planck 2018 cosmological parameters
    H0 = 100  # Hubble constant at z=0 in km/s/(Mpc/h)
    Omega_Lambda = 1.0 - Omega_m

    Hz = H0 * np.sqrt(Omega_m * (1 + redshift)**3 + Omega_Lambda)
    return Hz  # in km/s/(Mpc/h)


logmass_bins = np.linspace(11.5,15,17)
mass_bins = 10**logmass_bins
volum = 100**3  # (Mpc/h)^3
def hmf(data):
    counts, _ = np.histogram(data, bins=mass_bins)
    dn_dlogm = counts / volum / np.diff(logmass_bins)
    return dn_dlogm

def hist(data,bins):
    counts, _ = np.histogram(data, bins=bins)
    return counts

BoxSize = 100.0  # Mpc/h
def power_spectrum(pos, weight=None):
    pos = np.float32(pos)
    Ngrid = 128
    MAS = 'NGP'
    delta = np.zeros((Ngrid,Ngrid,Ngrid),dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, verbose=False, W=weight)
    delta /= np.mean(delta, dtype=np.float32);  delta -= 1.0
    Pk = PKL.Pk(delta, BoxSize, axis=2, MAS='None', threads=8, verbose=False)
    k = Pk.k3D
    Pk0 = Pk.Pk[:,0]
    Pk2 = Pk.Pk[:,1]
    return k, Pk0, Pk2

grid = 16
add_space_token = False
dim_tot = 8
start_token = 65
space_token = 66
end_token = 68
dim_prop = 5
nvocab = 64
xarray = np.arange(nvocab) * (BoxSize / nvocab / grid)
xarray = np.concatenate((xarray, [BoxSize/grid]))
bins_digitize = np.zeros((dim_prop, nvocab+1))
v_left = np.linspace(-1200, -100, 22, endpoint=False)
v_mid = np.linspace(-100,  100, 19, endpoint=False) 
v_right = np.linspace(100, 1200, 23, endpoint=True)
temp = np.linspace(11.5, 14.5, nvocab)
bins_digitize[0,1:-1] = 0.5*(temp[:-1] + temp[1:])  # Mass
bins_digitize[0,0] = 11.5
bins_digitize[0,-1] = 14.5
temp = np.concatenate((v_left, v_mid, v_right))
for i in range(1,4):
    bins_digitize[i,1:-1] = 0.5*(temp[:-1] + temp[1:])  # Velocities
    bins_digitize[i,0] = -1200.0
    bins_digitize[i,-1] = 1200.0
temp = np.linspace(1.0, 30.0, nvocab)
bins_digitize[4,1:-1] = 0.5*(temp[1:]+temp[:-1]) # concentration
bins_digitize[4,0] = 1.0
bins_digitize[4,-1] = 30.0

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
                        print(Nhalos_here, 'Nhalos_here is not an integer')
                        Nhalos_here = int(Nhalos_here)
                    #else:
                    if Nhalos_here > 0:
                            for jh in range(int(Nhalos_here)):
                                try:
                                    prop_all = np.zeros(dim_prop)        
                                    for jp in range(dim_prop):
                                        # noise_val = np.random.uniform(-delta_bin/2, delta_bin/2)
                                        bin_val_jp = sentence_here[ind_start_token + jh*ntokens_per_halo + 4 + jp]
                                        # try:
                                            # if jh == 0:
                                        #prop_all[jp] = prop_min[jp] + (prop_max[jp] - prop_min[jp]) * bins_digitize[bin_val_jp]
                                        prop_all[jp] = bins_digitize[jp, bin_val_jp]
                                            # else:
                                            #     prop_all[jp] = 0.0
                                        # except:
                                            # pass

                                    coord_x = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 1]] + (BoxSize/grid)*jx) % BoxSize
                                    coord_y = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 2]] + (BoxSize/grid)*jy) % BoxSize
                                    coord_z = (xarray[sentence_here[ind_start_token + jh*ntokens_per_halo + 3]] + (BoxSize/grid)*jz) % BoxSize
                                    pos_infer_all.append([coord_x, coord_y, coord_z])
                                            # print(sentence_here)
                                    prop_infer_all.append(prop_all)

                                except Exception as e:
                                   print(e)
                                pass
                else:           
                    print('End token not found')
                    pass
    #pos_infer_all = np.concatenate(pos_infer_all, axis=0)
    #prop_infer_all = np.concatenate(prop_infer_all, axis=0)
    return np.array(pos_infer_all), np.array(prop_infer_all)


def get_sim_number(filepath):
    filename = os.path.basename(filepath)
    number = int(filename.split('_')[3])
    return number

norm = colors.Normalize(vmin=0.1, vmax=0.6)
cmap = cm.get_cmap('viridis')

gen_dir = '/work/hdd/bdne/yzhang116/generates_36/'
files = sorted(glob.glob(os.path.join(gen_dir, "*snap3.npy")), key=get_sim_number)
all_param = np.loadtxt('/work/hdd/bdne/yzhang116/CAMEL_SAM_params.txt',usecols=(1),unpack=True)
simids = [get_sim_number(f) for f in files]
filenum = len(files)
mcrit = 10**11.5
fig, axs = plt.subplots(1,3, figsize=(15,5))
for i in range(filenum):
    simid = simids[i]
    Om = all_param[simid]
    print(f"SimID: {simid}, Omega_m: {Om}")
    data = np.load(files[i])
    data = np.delete(data, [1,2,3], axis=1)
    data = data.reshape((grid,grid,grid,-1))
    pos, prop = get_prop_pos(data)
    vel = prop[:,1:4]
    data = np.loadtxt('/work/hdd/bdne/yzhang116/halo_catalogs_11.5/halo_LH_%d.dat'%simid)
    pos_true = data[:, 1:4]
    v_true   = data[:, 4:7]
    mass     = data[:, 0]
    rs       = data[:, 7]
    index = np.where(mass > mcrit)[0]
    pos_true = pos_true[index]
    v_true = v_true[index]
    mass = mass[index]
    rs = rs[index]
    num_true = mass.shape[0]
    num_infer = prop.shape[0]
    '''
    num_min = min(num_infer, num_true)
    if num_true > num_infer:
        select_idx = random.sample(range(num_true), num_min)
        pos_true = pos_true[select_idx]
    else:
        select_idx = random.sample(range(num_infer), num_min)
        pos = pos[select_idx]
    '''

    #if num_true <= 10000:
    #    continue
    print(f"generate: {num_infer}, true: {num_true}")
    hmf1 = hmf(np.power(10, prop[:,0]))
    hmf2 = hmf(mass)
    axs[0].plot(logmass_bins[:-1], (hmf1/hmf2-1)*100, alpha=0.3, c=cmap(norm(Om)))

    k, Pk0, _ = power_spectrum(pos)#,weight=np.float32(prop[:,0]-11.5)**2)
    k1, Pk1, _ = power_spectrum(pos_true)#,weight=np.float32(np.log10(mass)-11.5)**2)
    Pk0 -= (BoxSize**3/num_infer)
    Pk1 -= (BoxSize**3/num_true)
    axs[1].plot(k, (Pk0/Pk1-1)*100, alpha=0.3, c=cmap(norm(Om)))
    
    pos = np.float32(pos)
    vel_z = np.float32(vel)
    pos_true = np.float32(pos_true)
    vz_true = np.float32(v_true)
    pos_rsd = pos.copy()
    pos_rsd_true  = pos_true.copy()
    
    RSL.pos_redshift_space(pos_rsd,vel_z, BoxSize, hubble(0.5, Om), 0.5, axis=2)
    RSL.pos_redshift_space(pos_rsd_true, vz_true, BoxSize, hubble(0.5, Om), 0.5, axis=2)

    k_rsd, Pk0_rsd, Pk02_rsd = power_spectrum(pos_rsd)#,weight=np.float32(prop[:,0]-11.5)**2)
    k1_rsd, Pk1_rsd, Pk12_rsd = power_spectrum(pos_rsd_true)#,weight=np.float32(np.log10(mass)-11.5)**2)
    Pk0_rsd -= (BoxSize**3/num_infer)
    Pk1_rsd -= (BoxSize**3/num_true)
    axs[2].plot(k_rsd, (Pk0_rsd/Pk1_rsd-1)*100, alpha=0.3, c=cmap(norm(Om)))
    


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

axs[2].axhline(1.0, color='k', linestyle='--')
axs[2].set_xlabel('k [h/Mpc]')
axs[2].set_ylabel('P_0 Percentage difference (%)')
axs[2].set_xscale('log')
axs[2].set_ylim(-50,50)
axs[2].grid()

plt.tight_layout()
plt.savefig('performance_summary.png')
