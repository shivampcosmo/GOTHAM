
import os
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
root = "/work/hdd/bdne/yzhang116/quijote/"

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
stepper = "bullfrog"
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
    path_ic = root + "LH_%d/ICs"%(sim_id)
    path_pm = root + "LH_%d/pm"%(sim_id)
    Om = 0.3175
    sigma8 = 0.834
    Oc = Om - Ob

    if not(os.path.exists(path_pm)):  os.system('mkdir %s'%path_pm)
    ic_delta = np.load(path_ic+"/IC_delta640.npy")
    ic_delta = jnp.array(ic_delta, dtype=jnp.float32)

    cosmo = dict(Omega_c=Oc,  # cold dark matter content
                 Omega_b=Ob,  # baryonic content (note: Disco-DJ so far only performs N-body simulations, no hydro; this is only used for the linear power spectrum!)
                 h=0.6711,  # dimensionless Hubble constant
                 n_s=0.9642,  # scalar spectral index
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
    X_sim = np.array(X_sim, dtype=np.float32)
    P_sim = np.array(P_sim, dtype=np.float32)
    X_sim = np.reshape(X_sim, (-1,3))
    P_sim = np.reshape(P_sim, (-1,3))

    print("Saving to %s"%(path_pm))

    np.save(path_pm+"/pos_LH%d_z05.npy"%sim_id,X_sim)
    np.save(path_pm+"/vel_LH%d_z05.npy"%sim_id,P_sim)

    print("Simulation %d done."%(sim_id))


    del dj, ic_delta, X_sim, P_sim
    gc.collect()

    return


for i in range(1):
    run_one_simulation(i)
