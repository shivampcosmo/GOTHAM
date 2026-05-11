import numpy as np
import matplotlib.pyplot as plt

data = np.load(f'/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits.npz')
emb = data['wte']          # (vocab_size, n_embd)

'''
emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
ref_index = 20
ref_norm = emb_norm[ref_index]                       # (D,)

# cosine similarity: (V,)
similarity = emb_norm @ ref_norm
plt.figure()
plt.plot(similarity, marker='o')
plt.xlabel("Token index")
plt.ylabel("Cosine similarity")
plt.savefig("results/embedding_20.png", dpi=300)
plt.close()
'''

# this is for vit
pos_emb = data['pos_embed'][0]          # (64, n_embd)
print(pos_emb.shape)
pos_emb_norm = pos_emb / np.linalg.norm(pos_emb, axis=1, keepdims=True)
ref_index = 0
ref_norm = pos_emb_norm[ref_index]                       # (D,)
similarity = pos_emb_norm @ ref_norm
similarity = similarity[:64]
similarity = similarity.reshape((4,4,4))

fig, axs = plt.subplots(2,2, figsize=(12,12))
for i in range(2):
    for j in range(2):
        idx = i*2 + j
        im = axs[i,j].imshow(similarity[idx][:4,:4], cmap='viridis', vmin=0, vmax=1)
        axs[i,j].set_title(f"X={idx}", fontsize=18)
        axs[i,j].set_xticks(range(4))
        axs[i,j].set_yticks(range(4))
        axs[i,j].set_xlabel("Z", fontsize=18)
        axs[i,j].set_ylabel("Y", fontsize=18)
        axs[i,j].set_xticklabels(range(4), fontsize=18)
        axs[i,j].set_yticklabels(range(4), fontsize=18)

fig.tight_layout()
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=18)
plt.savefig(f"pos_embedding.png", dpi=300)
