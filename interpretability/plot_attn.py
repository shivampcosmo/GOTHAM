import numpy as np
import matplotlib.pyplot as plt

cross = np.load('/work/hdd/bdne/yzhang116/cp_visualize/cross_avg_layer_head_cosmo_66_5halos.npy')
print(cross.shape) 
'''
fig, axs = plt.subplots(1, 1)
im = axs.imshow(cross[6:], cmap='viridis', vmin=0, vmax=0.1)
axs.set_xticks([])
axs.set_yticks([])
fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, shrink=0.6)
fig.savefig('cross_attn.png', dpi=300, bbox_inches='tight')
'''
self_att = np.load('/work/hdd/bdne/yzhang116/cp_visualize/self_avg_laryer_head_cosmo_66.npy')
print(self_att.shape)
nhalo = 5
nprop = 8
fig, axs = plt.subplots(1, 1)
im = axs.imshow(self_att[7:7+nhalo*nprop, :7+nhalo*nprop], cmap='viridis', vmin=0, vmax=0.15)
axs.set_xticks([])
axs.set_yticks([])
fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, shrink=1)
fig.savefig('self_attn.png', dpi=300, bbox_inches='tight')