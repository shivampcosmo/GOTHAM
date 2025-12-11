import numpy as np
from mpi4py import MPI
import os

###### MPI DEFINITIONS ######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

points = 1000

numbers = np.where(np.arange(points)%nprocs==myrank)[0]

for sim_id in numbers:
    #nfiles = 8
    nfiles = 1
    try:
        #path = f'/work/hdd/bdne/yzhang116/ICsSAM/cLH{sim_id:d}/ICs/'
        #path = f'/work/hdd/bdne/yzhang116/CV/CV_{sim_id:d}/ICs/'
        path = f'/work/hdd/bdne/yzhang116/quijote/LH_%d/ICs/'%sim_id
        idx = []
        amp, phase = [], []
        print("\nConvert for path: ", path)

        for i in range(nfiles):
            # read a particular coordinate file
            f_in = path + 'Coordinates_ptype_1.%d'%i
            f = open(f_in, 'rb')
            Nfiles = np.fromfile(f, dtype=np.int32, count=1)[0] #Number of coordinate subfiles  
            Nmesh  = np.fromfile(f, dtype=np.int32, count=1)[0] #Nmesh size
            Nx     = np.fromfile(f, dtype=np.int32, count=1)[0] #slab offset (not used)
            coordinates = np.fromfile(f, dtype=np.int64, count=-1)
            kx = (coordinates//(Nmesh//2 + 1))//Nmesh
            ky = (coordinates//(Nmesh//2 + 1))%Nmesh
            kz = (coordinates%(Nmesh//2 + 1))%Nmesh
            kk = np.array([kx, ky, kz]).T
            idx.append(kk)

            f_in = path + 'Amplitudes_ptype_1.%d'%i
            f = open(f_in,'rb')
            Nfiles = np.fromfile(f, dtype=np.int32, count=1)[0] #Number of coordinate subfiles  
            Nmesh  = np.fromfile(f, dtype=np.int32, count=1)[0] #Nmesh size
            Nx     = np.fromfile(f, dtype=np.int32, count=1)[0] #slab offset (not used)
            aa = np.fromfile(f, dtype=np.float32, count=-1)
            amp.append(aa)
            f_in = path + 'Phases_ptype_1.%d'%i
            f = open(f_in,'rb')
            Nfiles = np.fromfile(f, dtype=np.int32, count=1)[0] #Number of coordinate subfiles  
            Nmesh  = np.fromfile(f, dtype=np.int32, count=1)[0] #Nmesh size
            Nx     = np.fromfile(f, dtype=np.int32, count=1)[0] #slab offset (not used)
            phase.append(np.fromfile(f, dtype=np.float32, count=-1))
    except:
        pass

    idx = np.concatenate(idx)
    amp = np.concatenate(amp)
    phase = np.concatenate(phase)

    val = amp*np.exp(1j*phase)
    cmesh = val.reshape(Nmesh, Nmesh, kz.max()+1)

    bs=100     #Mpc/h
    mesh = np.fft.irfftn(cmesh, norm='ortho') * Nmesh**1.5
    #np.save(path+'IC_delta128_cLH%d.npy'%sim_id, mesh.astype(np.float32))
    np.save(path+'IC_delta640.npy', mesh.astype(np.float32))


    for i in range(nfiles):
        os.remove(path + 'Coordinates_ptype_1.%d'%i)
        os.remove(path + 'Amplitudes_ptype_1.%d'%i)
        os.remove(path + 'Phases_ptype_1.%d'%i)
