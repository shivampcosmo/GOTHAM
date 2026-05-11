import dill
import sys, os, glob
import numpy as np 
import MAS_library as MASL
from galactic_wavelets.scattering_operator import ScatteringOp
import torch


def get_gal_mesh(pos, boxsize=1000., los=[1,0,0], grid=256, MAS='NGP'):
        mesh = np.zeros((grid, grid, grid), dtype=np.float32)
        MASL.MA(pos, mesh, boxsize, MAS)
        mesh /= np.mean(mesh, dtype=np.float32)
        mesh -= 1.0
        return mesh

def get_wavelets_all_hods(isim, noise, wst_op1, wst_op2, nhod_LH_samp=10, grid1=128, grid2=192):
    gal_pos = dill.load(open(f'/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod/galaxy_rsd_pos_noise/Galaxy_pos_galnoise_{noise:.1f}_LH_{isim}.dill', 'rb'))
    #gal_pos = dill.load(open(f'/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod/galaxy_rsd_pos_noise/Galaxy_pos_noise_{noise:.1f}_LH_{isim}.dill', 'rb'))
    saved_j = {'isim': isim}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for ihod in range(nhod_LH_samp):
        saved_j[f'theta_hod_{ihod}'] = gal_pos[f'theta_hod_{ihod}']
        saved_j[f'theta_cosmo_{ihod}'] = gal_pos[f'theta_cosmo_{ihod}']
        saved_j[f'galsum_mock_{ihod}'] = gal_pos[f'galsum_mock_{ihod}']
        saved_j[f'galsum_truth_{ihod}'] = gal_pos[f'galsum_truth_{ihod}']

        pos_mock = gal_pos[f'pos_rsd_mock_{ihod}']
        pos_truth = gal_pos[f'pos_rsd_truth_{ihod}']

        mesh1_mock = get_gal_mesh(pos_mock, grid=grid1)
        mesh1_truth = get_gal_mesh(pos_truth, grid=grid1)
        mesh2_mock = get_gal_mesh(pos_mock, grid=grid2)
        mesh2_truth = get_gal_mesh(pos_truth, grid=grid2)

        df = torch.from_numpy(mesh1_truth).to(device)
        s0_truth, s1_truth, s2_truth = wst_op1(df)
        s0_truth = s0_truth.cpu().numpy()
        s1_truth = s1_truth.cpu().numpy().flatten()
        s2_truth = s2_truth.cpu().numpy().flatten()

        saved_j[f'rsd_s0_truth_{ihod}_{grid1:d}'] = s0_truth
        saved_j[f'rsd_s1_truth_{ihod}_{grid1:d}'] = s1_truth
        saved_j[f'rsd_s2_truth_{ihod}_{grid1:d}'] = s2_truth

        df = torch.from_numpy(mesh2_truth).to(device)
        s0_truth, s1_truth, s2_truth = wst_op2(df)      
        s0_truth = s0_truth.cpu().numpy()
        s1_truth = s1_truth.cpu().numpy().flatten()
        s2_truth = s2_truth.cpu().numpy().flatten()
        saved_j[f'rsd_s0_truth_{ihod}_{grid2:d}'] = s0_truth
        saved_j[f'rsd_s1_truth_{ihod}_{grid2:d}'] = s1_truth
        saved_j[f'rsd_s2_truth_{ihod}_{grid2:d}'] = s2_truth  

        df = torch.from_numpy(mesh1_mock).to(device)
        s0_mock, s1_mock, s2_mock = wst_op1(df)
        s0_mock = s0_mock.cpu().numpy()
        s1_mock = s1_mock.cpu().numpy().flatten()
        s2_mock = s2_mock.cpu().numpy().flatten()
        saved_j[f'rsd_s0_mock_{ihod}_{grid1:d}'] = s0_mock
        saved_j[f'rsd_s1_mock_{ihod}_{grid1:d}'] = s1_mock
        saved_j[f'rsd_s2_mock_{ihod}_{grid1:d}'] = s2_mock

        df = torch.from_numpy(mesh2_mock).to(device)
        s0_mock, s1_mock, s2_mock = wst_op2(df)
        s0_mock = s0_mock.cpu().numpy()
        s1_mock = s1_mock.cpu().numpy().flatten()
        s2_mock = s2_mock.cpu().numpy().flatten()
        saved_j[f'rsd_s0_mock_{ihod}_{grid2:d}'] = s0_mock
        saved_j[f'rsd_s1_mock_{ihod}_{grid2:d}'] = s1_mock
        saved_j[f'rsd_s2_mock_{ihod}_{grid2:d}'] = s2_mock
        
    dill.dump(saved_j, open(f'/work/hdd/bdne/yzhang116/quijote_galaxy/WST_all/WST_J6Q2_NGP_galnoise_{noise:.1f}_galaxy_LH_{isim}.dill', 'wb'))

    return 

def get_sim_number(filename):
    base = os.path.basename(filename)
    number_str = base.split('_')[-1].replace('.dill', '')
    return int(number_str)

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"Using device: {device}", flush=True)
    grid1 = 128 
    J = 6
    Q = 2
    kc = np.pi/2.
    moments = [1/2, 1, 2]
    wst_op1 = ScatteringOp(torch.Size([grid1, grid1, grid1]), J, Q,
                            moments=moments,
                            kc=kc,
                            scattering=True,
                            device=device,
                            los = [1,0,0],
                            use_mp=False)
    grid2 = 192
    wst_op2 = ScatteringOp(torch.Size([grid2, grid2, grid2]), J, Q,
                            moments=moments,
                            kc=kc,
                            scattering=True,
                            device=device,
                            los = [1,0,0],
                            use_mp=False)
    '''
    for isim in simids:
        outfile = f'/work/hdd/bdne/yzhang116/quijote_galaxy/WST_all/WST_galaxy_LH_{isim}.dill'
        if os.path.exists(outfile):
            print(f"Skip {isim}, file already exists.", flush=True)
            continue
        get_wavelets_all_hods(isim,wst_op1,wst_op2,nhod_LH_samp=10, grid=grid)
    '''
    for isim in range(start, end):
        get_wavelets_all_hods(isim,noise,wst_op1,wst_op2,nhod_LH_samp=10, grid1=grid1, grid2=grid2)