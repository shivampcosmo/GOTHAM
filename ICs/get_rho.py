import numpy as np
import os, sys
import MAS_library as MASL


def save_cic_densities(ji):
    # try:
    root_in = '/mnt/ceph/users/spandey/discodj_runs/LH/%d/pm' % ji
    root_out = '/mnt/ceph/users/spandey/discodj_runs/rhog_LH/%d/' % ji
    grid         = 512
    
    os.makedirs(root_out, exist_ok=True)
    fname_out_rho = root_out+"/rho_grid_%d_LH%d_z05.npy"%(grid, ji)
    fname_out_vel = root_out+"/vel_grid_%d_LH%d_z05.npy"%(grid, ji)
    if os.path.exists(fname_out_rho) and os.path.exists(fname_out_vel):
        return
    else:
        
        BoxSize = 1000.0 #Mpc/h ; size of box
        BoxSize = 1000.0    
        pos = np.load(root_in + '/pos_LH%d_z05.npy' % ji)
        pvel = np.load(root_in + '/vel_LH%d_z05.npy' % ji)                           
        
        # pos = np.array(df['Position'], dtype=np.float64)
        rho_m_orig = np.zeros((grid,grid,grid), dtype=np.float32)
        MASL.MA(np.float32(pos), rho_m_orig, BoxSize, 'CIC', verbose=False)
        

        mom_m_x_orig = np.zeros((grid,grid,grid), dtype=np.float32)                
        MASL.MA(np.float32(pos), mom_m_x_orig, BoxSize, 'CIC', verbose=False, W=pvel[:,0].astype(np.float32))
        mom_m_y_orig = np.zeros((grid,grid,grid), dtype=np.float32)                
        MASL.MA(np.float32(pos), mom_m_y_orig, BoxSize, 'CIC', verbose=False, W=pvel[:,1].astype(np.float32))
        mom_m_z_orig = np.zeros((grid,grid,grid), dtype=np.float32)                
        MASL.MA(np.float32(pos), mom_m_z_orig, BoxSize, 'CIC', verbose=False, W=pvel[:,2].astype(np.float32))                

        vel_m_x_orig = mom_m_x_orig/rho_m_orig
        vel_m_y_orig = mom_m_y_orig/rho_m_orig
        vel_m_z_orig = mom_m_z_orig/rho_m_orig

        # at voxels with non-finite values, set velocities to zero:
        vel_m_x_orig[~np.isfinite(vel_m_x_orig)] = 0.0
        vel_m_y_orig[~np.isfinite(vel_m_y_orig)] = 0.0
        vel_m_z_orig[~np.isfinite(vel_m_z_orig)] = 0.0

        # divide out by 1000.:
        vel_m_x_orig /= 100.
        vel_m_y_orig /= 100.
        vel_m_z_orig /= 100.

        vel_m_all_orig = np.stack((vel_m_x_orig, vel_m_y_orig, vel_m_z_orig), axis=0).astype(np.float16)
        rho_m_orig = (rho_m_orig / 100.).astype(np.float16)
        print(np.amin(vel_m_all_orig), np.amax(vel_m_all_orig))
        print(np.amax(rho_m_orig), np.min(rho_m_orig))
        np.save(root_out+"/rho_grid_%d_LH%d_z05.npy"%(grid, ji), rho_m_orig)
        np.save(root_out+"/vel_grid_%d_LH%d_z05.npy"%(grid, ji), vel_m_all_orig)
        return

sim_id = int(sys.argv[1])

save_cic_densities(sim_id)
