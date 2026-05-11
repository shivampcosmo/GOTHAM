import numpy as np
import matplotlib.pyplot as plt
import sys

data = np.load(f'/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits.npz')
logits = data['logits_100'] 
sentences = data['sentence']
index = np.where(sentences[:,6]>= 3)[0] # Find the index of the first halo
print(index)
print(sentences[index,6])
index = int(sys.argv[1]) # 5 halos
nhalo = 3
nprop = 8
logits_sel = logits[index]  # (L-1, vocab_sizes)
sentences_sel = sentences[index]  # (L,)
labels = [r'$x$', r'$y$', r'$z$', 'mass', r'$v_x$', r'$v_y$', r'$v_z$', 'concentration']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
prop = np.exp(logits_sel) / np.exp(logits_sel).sum(axis=-1, keepdims=True)  # (L-1, vocab_size)
prop = prop[6:6+nhalo*nprop].reshape((nhalo, 8, -1))  # (nhalo, 8, vocab_size)
sentences_sel = sentences_sel[7:7+nhalo*nprop].reshape((nhalo, 8))  # (nhalo, 8)
fig, axs = plt.subplots(2,4, figsize=(20, 10))
axs = axs.flatten()
for i in range(nhalo):
    for j in range(nprop):
        idx = i*nprop + j
        im = axs[j].plot(prop[i,j,:], color=colors[i], label=f"Halo {i+1}" if j==3 else None)
        axs[j].vlines(x=sentences_sel[i,j], ymin=0, ymax=np.max(prop[:,j,:]), color=colors[i], linestyle='dashed')
        axs[j].set_title(f"{labels[j]}", fontsize=18)
for i in range(8):
    axs[i].set_xlabel("Token index", fontsize=18)
for i in range(3):
    axs[i].set_xlim(0, 41)
axs[0].set_ylabel("Probability", fontsize=18)
axs[4].set_ylabel("Probability", fontsize=18)

fig.tight_layout()
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar_ax.axis('off')
axs[3].legend(loc='upper right', fontsize=18)
plt.savefig(f"logits_halo_{index}.png", dpi=300)