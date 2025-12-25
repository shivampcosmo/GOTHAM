import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# preallocate 95% of the GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
import jax
from jax import config
import numpy as np
from discodj import DiscoDJ
import jax.numpy as jnp
from jax._src.lib import xla_client
import sys
import gc

devices = jax.devices()
device = "gpu" if np.any([d.platform == "gpu" for d in devices]) else "cpu"
# device = "cpu"
root = "/mnt/ceph/users/spandey/discodj_runs/LH/"

dim = 3
precision = "single"
# Define the boxsize and resolution
boxsize = 1000.0  # in Mpc/h
res = 640  # the particles live on a Lagrangrian grid of resolution (res)^dim
factor = 2
n_order = 2
a_ic = 1./128.
numsteps = 20  # number of steps to be performed
a_end = 1./1.5
Ob = 0.049
# stepper = "bullfrog"
stepper = "fastpm"
method = "pm"
res_pm = factor * res
time_var = "D"
antialias = 0
grad_kernel_order = 4
laplace_kernel_order = 0
worder = 2
n_resample = 1
deconvolve = False
nlpt_order_ics = n_order
chunk_size = None




def run_one_simulation(sim_id):
        path_ic = root + "%d/ICs"%(sim_id)
        path_pm = root + "%d/pm"%(sim_id)
        os.makedirs(path_pm, exist_ok=True)

        path_sim_pos = path_pm+"/pos_LH%d_z05.npy"%sim_id
        path_sim_vel = path_pm+"/vel_LH%d_z05.npy"%sim_id
        # check if the file exists, else print the number:
        if not os.path.exists(path_sim_pos) or not os.path.exists(path_sim_vel):
                print("File missing for sim_id: ", sim_id)


                # read Cosmo_params.dat file to get cosmological parameters:
                cosmo_all = np.loadtxt(path_ic + "/Cosmo_params.dat", delimiter=' ')



                Om = cosmo_all[0]
                Ob = cosmo_all[1]
                h = cosmo_all[2]
                ns = cosmo_all[3]
                sigma8 = cosmo_all[4]
                Oc = Om - Ob

                # if not(os.path.exists(path_pm)):  os.system('mkdir %s'%path_pm)
                ic_delta = np.load(path_ic+"/IC_delta640.npy")
                ic_delta = jnp.array(ic_delta, dtype=jnp.float32)

                cosmo = dict(Omega_c=Oc,  # cold dark matter content
                                Omega_b=Ob,  # baryonic content (note: Disco-DJ so far only performs N-body simulations, no hydro; this is only used for the linear power spectrum!)
                                h=h,  # dimensionless Hubble constant
                                n_s=ns,  # scalar spectral index
                                sigma8=sigma8  # amplitude of matter density fluctuations at a scale of 8 Mpc/h
                                )

                dj = DiscoDJ(dim=dim, res=res, device=device, precision=precision, boxsize=boxsize, cosmo=cosmo)

                # Compute the linear power spectrum
                dj = dj.with_timetables()
                Dplus_aic = np.interp(jnp.log10(a_ic), jnp.log10(dj.cosmo._timetables['a']), dj.cosmo._timetables['Dplus'])

                dj = dj.with_external_ics(delta = ic_delta/Dplus_aic)
                dj = dj.with_lpt(n_order=n_order, try_to_jit=True)

                X_sim, P_sim, _ = dj.run_nbody(
                        a_ini=a_ic, a_end=a_end, n_steps=numsteps, res_pm=res_pm,
                        time_var=time_var, stepper=stepper, method=method,
                        antialias=antialias, grad_kernel_order=grad_kernel_order,
                        laplace_kernel_order=laplace_kernel_order,
                        nlpt_order_ics=nlpt_order_ics, n_resample=n_resample,
                        deconvolve=deconvolve, return_displacement=False,
                        chunk_size=chunk_size)
                # convert to km/s
                P_sim = P_sim / a_end * 100.
                X_sim = np.array(X_sim, dtype=np.float16)
                P_sim = np.array(P_sim, dtype=np.float16)
                X_sim = np.reshape(X_sim, (-1,3))
                P_sim = np.reshape(P_sim, (-1,3))

                print("Saving to %s"%(path_pm))
                # np.savez_compressed(path_pm + "/sim_LH%d_z05.npz" % (sim_id), pos=X_sim, vel=P_sim)
                np.save(path_pm+"/pos_LH%d_z05.npy"%sim_id,X_sim)
                np.save(path_pm+"/vel_LH%d_z05.npy"%sim_id,P_sim)

                print("Simulation %d done."%(sim_id))


                del dj, ic_delta, X_sim, P_sim
                gc.collect()
                jax.clear_caches()
                return
        else:
                return

# clear up jax caches
jax.clear_caches()
gc.collect()

sim_id = int(sys.argv[1])

run_one_simulation(sim_id)

