import numpy as np
import matplotlib.pyplot as plt

# MNRAS journal style: serif/Times + STIX math fonts
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

data = np.load(f'/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits.npz')

# positional embeddings from ViT
pos_emb = data['pos_embed'][0]          # (64, n_embd)
pos_emb_norm = pos_emb / np.linalg.norm(pos_emb, axis=1, keepdims=True)
ref_index = 0
ref_norm = pos_emb_norm[ref_index]      # (D,)
similarity = pos_emb_norm @ ref_norm
similarity = similarity[:64].reshape((4, 4, 4))

# MNRAS single-column width ~3.4 in
fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.2))

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        im = axs[i, j].imshow(similarity[idx][:4, :4], cmap='Blues', vmin=0, vmax=1)
        axs[i, j].set_title(f'$x={idx}$')
        axs[i, j].set_xticks(range(4))
        axs[i, j].set_yticks(range(4))
        axs[i, j].set_xticklabels(range(4))
        axs[i, j].set_yticklabels(range(4))
        if i == 1:
            axs[i, j].set_xlabel('$z$')
        if j == 0:
            axs[i, j].set_ylabel('$y$')

fig.tight_layout()
fig.subplots_adjust(right=0.86, wspace=0.15, hspace=0.35)
cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Cosine similarity')
cbar.ax.tick_params(labelsize=7)

plt.savefig('pos_embedding.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pos_embedding.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved pos_embedding.pdf / .png")
