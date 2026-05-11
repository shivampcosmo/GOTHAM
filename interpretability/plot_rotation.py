import numpy as np
import matplotlib.pyplot as plt
import os

print('Loading data...')
data = np.load('/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits_blocks.npz')
HALO_DIR       = '/work/hdd/bdne/yzhang116/halo_sentence_full_quijote'
halo_path   = os.path.join(HALO_DIR, f'halo_sentence_LH_66.npy')
halo_sentence = np.load(halo_path) # (N_total_subbox, L)
halo_sentence = halo_sentence[:100]  # (100, L)
print(halo_sentence[2:10,6:16])
print(f'halo_sentence shape: {tuple(halo_sentence.shape)}')
wte = data['wte']  # (vocab_size, n_embd)
x_pre = data['x_pre'][:,7:] # (B, L-1, n_embd)
B = x_pre.shape[0]
sentences = data['sentence']
nhalos = sentences[:,6]  # (B)
nprop = 8
x_after = np.zeros((9,x_pre.shape[0],x_pre.shape[1],x_pre.shape[2]))
similarity = np.zeros((2,9))
similarity_std = np.zeros((2,9))
x_after[0] = x_pre
for i in range(8):
    x_after[i+1] = data['x_block_%d'%i][:,7:] # (B, L-1, n_embd)
#x_after[8] = data['x_lnf'][:,7:] # (B, L-1, n_embd)
print(f'x_pre shape: {x_pre.shape}')
prop_emb = data['wprope']
halos_emb = data['whe']
embs = np.zeros((B, nprop*36, x_pre.shape[2]))
for i in range(B):
    embs[i] = wte[halo_sentence[i,6:6+36*nprop]]
print(f'prop_emb shape: {prop_emb.shape}')
#embs = np.zeros((nprop*36, x_pre.shape[2]))
#for i in range(36):
#    print(f'Processing halo {i+1}/36')
#    for j in range(nprop):
#        embs[i*nprop+j] = prop_emb[j] + halos_emb[i]
#x_pre = np.expand_dims(embs, axis=0)  # (B, max_nhalo*nprop, n_embd)
#x_pre = np.repeat(x_pre, B, axis=0)  # (B, max_nhalo*nprop, n_embd)

x_pre = embs
#x_pre = x_pre - embs[None, :, :]
#x_after = x_after - embs[None, None, :, :]
x_pre_norm = x_pre / np.linalg.norm(x_pre, axis=-1, keepdims=True)
x_after_norm = x_after / np.linalg.norm(x_after, axis=-1, keepdims=True)
# mask: valid token positions per sample, shape (B, L-1)
k_idx = np.arange(x_pre.shape[1])
max_k = (nprop * nhalos).astype(int)
mask = k_idx[None, :] < max_k[:, None]          # (B, L-1)

# x_after[i,j,k] · x_pre[j,k]  →  (8, B, L-1)
dots = np.einsum('ibld,bld->ibl', x_after_norm, x_pre_norm)
similarity[0] = (dots * mask[None]).sum(axis=(1, 2)) / mask.sum()
similarity_std[0] = np.sqrt(((dots - similarity[0][:, None, None])**2 * mask[None]).sum(axis=(1, 2)) / mask.sum())

# x_after[i,j,k] · x_pre[j,k+1]  →  (8, B, L-2)
dots_shift = np.einsum('ibld,bld->ibl', x_after_norm[:, :, :-1, :], x_pre_norm[:, 1:, :])
mask_shift = mask[:, :-1]                        # (B, L-2)
similarity[1] = (dots_shift * mask_shift[None]).sum(axis=(1, 2)) / mask_shift.sum()
similarity_std[1] = np.sqrt(((dots_shift - similarity[1][:, None, None])**2 * mask_shift[None]).sum(axis=(1, 2)) / mask_shift.sum())

plt.figure()
x_axis = np.arange(similarity.shape[1])
line0, = plt.plot(x_axis, similarity[0], marker='o', label='Similarity to X')
plt.fill_between(x_axis, similarity[0] - similarity_std[0], similarity[0] + similarity_std[0], alpha=0.3, color=line0.get_color())
line1, = plt.plot(x_axis, similarity[1], marker='o', label='Similarity to Y')
plt.fill_between(x_axis, similarity[1] - similarity_std[1], similarity[1] + similarity_std[1], alpha=0.3, color=line1.get_color())
plt.xlabel("Block index")
plt.ylabel("Cosine similarity")
plt.title("Cosine similarity of block outputs to value embeddings")
plt.legend()
plt.savefig("block_similarity_prop.png", dpi=300)




