"""visualize_cross_attn_avg.py

Cross-attention (decoder ← encoder) averaged over ALL sub-boxes and in different cosmology.

Because different sub-boxes have different spatial patch layouts, encoder patch
columns cannot be averaged directly.  Instead, for each sub-box the N_patches
patch tokens are re-ordered by their local density (channel DENSITY_CH),
highest-to-lowest.  The 5 fixed cosmology tokens stay at the front unchanged.

  x-axis: [C0 C1 C2 C3 C4 | patch@density-rank-0 … patch@density-rank-N-1]
  y-axis: decoder token position 0 → end_token -1
           (averaged per position; token i is only averaged over sub-boxes
            that contain at least i+1 halo tokens)

Output: 8 PNG figures (one per decoder layer), each with 3×4 panels (12 heads).

Usage (compute node):
    cd /u/yzhang116/NN/quijote
    python visualize_cross_attn.py
"""

import sys, os, math
sys.path.insert(0, '/u/yzhang116/NN/quijote')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle as pk
from model_enc_dec_cos_fast import HaloDecoderModel

# ============================================================
# Config — edit here
# ============================================================
CHECKPOINT     = '/work/hdd/bdne/yzhang116/checkpoints_quijote/1872236/checkpoint_cos_cross_entropy_epoch_0_step_7050_embed_384_batch_800_lrmax_1e-05_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt'
DMO_DIR        = '/work/nvme/bdne/yzhang116/quijote_fields/train/'
HALO_DIR       = '/work/nvme/bdne/yzhang116/quijote_halos/train/'
#DMO_DIR        = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3'
#HALO_DIR       = '/work/hdd/bdne/yzhang116/halo_sentence_full_quijote'
META_PATH      = '/work/nvme/bdne/yzhang116/quijote_halos/sentence_params.pkl'
OUT_DIR        = '/u/yzhang116/NN/plots/attention_avg/'
SIM_ID         = None   # None → random cosmology; int → specific sim (e.g. 0)
DENSITY_CH     = 10     # 0-indexed DM field channel for density ranking
N_COSMO_TOKENS = 5      # ViT encoder prepends 5 fixed cosmology tokens before patches
N_BATCH        = 500    # sub-boxes per forward pass; reduce if OOM
SEED         = 66
# ============================================================

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#ctx    = torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16)

# ---- Metadata ----
meta_f      = pk.load(open(META_PATH, 'rb'))
pad_token   = int(meta_f['pad_token'])
end_token   = int(meta_f['end_token'])
start_token = int(meta_f['start_token'])
print(f'pad_token={pad_token}  end_token={end_token}  start_token={start_token}')

# ---- Model ----
ckpt       = torch.load(CHECKPOINT, map_location=DEVICE)
HaloConfig = ckpt['config']
HaloConfig['device'] = DEVICE
HaloConfig['flash']  = True   # manual attention — no EFFICIENT_ATTENTION backend needed
model = HaloDecoderModel(HaloConfig).to(DEVICE)
sd    = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
model.load_state_dict(sd, strict=False)
model.eval()

n_layers = HaloConfig['n_layer']
n_heads  = HaloConfig['n_head']

# ---- Pick simulation ----
all_dm_files = sorted(
    [f for f in os.listdir(DMO_DIR) if f.endswith('.npy')],
    key=lambda f: int(f.split('_')[-1].split('.')[0])
)
sim_numbers = [int(f.split('_')[-1].split('.')[0]) for f in all_dm_files]
chosen_sim  = np.random.choice(sim_numbers) if SIM_ID is None else SIM_ID
dm_path     = os.path.join(DMO_DIR,  f'fields_{chosen_sim}.npy')
halo_path   = os.path.join(HALO_DIR, f'halos_{chosen_sim}.npy')
print(f'Sim {chosen_sim}:  {dm_path}')

DM_all   = np.load(dm_path)    # (N_subbox, D, H, W, C)
gal_all  = np.load(halo_path)  # (N_subbox, L)
#N_subbox = DM_all.shape[0]
N_subbox = 500
L_seq    = gal_all.shape[1] - 1   # length of X (= gal[:, :-1])
D_sp     = DM_all.shape[1]        # spatial side length of DM field
print(f'N_subbox={N_subbox}  L_seq={L_seq}  D_sp={D_sp}')

# ============================================================
# Cross-attention hooks
# ============================================================
cross_attn_store = []


def make_cross_hook():
    def hook(module, args, kwargs, output):
        xd = args[0]
        xe = kwargs.get('xe')
        if xe is None:
            cross_attn_store.append(None)
            return
        B, Td, C = xd.shape
        _, Te, _  = xe.shape
        n_head    = module.n_head
        head_dim  = C // n_head
        dev = xd.device.type
        with torch.no_grad(), torch.amp.autocast(device_type=dev, enabled=False):
            q = module.q_attn(xd.float()).view(B, Td, n_head, head_dim).transpose(1, 2)
            k = module.k_attn(xe.float()).view(B, Te, n_head, head_dim).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            att = torch.softmax(att, dim=-1)
        cross_attn_store.append(att.cpu())               # (B, n_head, Td, Te) float32
    return hook


hooks = []
for block in model.transformer.h:
    hooks.append(block.crossattn.register_forward_hook(make_cross_hook(), with_kwargs=True))

# ============================================================
# Accumulation arrays (initialised lazily after first forward pass)
# ============================================================
cross_sum        = None   # (n_layers, n_heads, L_seq, Te)   float64
cross_cnt        = None   # (L_seq,)  — same for all layers  float64

Te               = None
N_patches        = None
nd               = None
patch_size_field = None   # spatial voxels per patch side

# ============================================================
# Main loop — N_BATCH sub-boxes per forward pass
# ============================================================
for batch_start in range(0, N_subbox, N_BATCH):
    batch_end = min(batch_start + N_BATCH, N_subbox)
    B = batch_end - batch_start
    print(f'  sub-boxes {batch_start + 1}–{batch_end}/{N_subbox}', end='\r', flush=True)

    DM  = torch.tensor(DM_all[batch_start:batch_end], dtype=torch.float32, device=DEVICE)
    DM  = DM.moveaxis(-1, 1) #.to(torch.bfloat16)             # (B, C, D, H, W)
    gal = torch.tensor(gal_all[batch_start:batch_end], dtype=torch.long, device=DEVICE)

    X = gal[:, :-1]                                          # (B, L_seq)
    Y = gal[:, 1:].clone()
    Y[:, :5] = pad_token
    pad_mask      = torch.logical_not(X != pad_token)
    masked_logits = torch.zeros(pad_mask.shape, device=DEVICE, dtype=torch.float32)
    MASK          = masked_logits.masked_fill(pad_mask, float('-inf'))[:, None, :]  # (B, 1, L_seq)

    # valid_mask: True for query positions strictly before end_token (end_token excluded)
    has_end       = (X == end_token).any(dim=1)                        # (B,)
    raw_end       = (X == end_token).int().argmax(dim=1)               # (B,) first end_token idx
    end_positions = torch.where(has_end, raw_end,
                                torch.full_like(raw_end, X.shape[1])) # (B,)
    positions     = torch.arange(X.shape[1], device=DEVICE)[None, :]  # (1, L_seq)
    valid_mask    = positions < end_positions[:, None]                  # (B, L_seq) bool

    # Density sort for all B samples at once (float32 for mean)
    dm_ch = DM[:, DENSITY_CH].cpu().float()   # (B, D, H, W)

    # Forward pass
    cross_attn_store.clear()
    with torch.no_grad():
        #with ctx:
            _ = model(X, DM, maskd=MASK, targets=Y)
            
    # Initialise accumulators once we know Te from the hook output
    if cross_sum is None:
        Te               = cross_attn_store[0].shape[3]   # att is (B, n_head, Td, Te)
        N_patches        = Te - N_COSMO_TOKENS
        nd               = round(N_patches ** (1 / 3))
        patch_size_field = D_sp // nd
        print(f'\nTe={Te}  N_patches={N_patches}  nd={nd}  '
              f'patch_size_field={patch_size_field}')
        cross_sum = np.zeros((n_layers, n_heads, L_seq, Te), dtype=np.float64)
        cross_cnt = np.zeros(L_seq, dtype=np.float64)    # (L_seq,) — same for all layers

    # Density ranking for all B sub-boxes: (B, N_patches), descending
    patch_density  = (
        dm_ch
        .view(B, nd, patch_size_field, nd, patch_size_field, nd, patch_size_field)
        .mean(dim=(2, 4, 6))   # (B, nd, nd, nd)
        .view(B, -1)           # (B, N_patches)
    )
    sort_idx_batch = torch.argsort(patch_density, descending=True, dim=1)  # (B, N_patches)

    # Vectorized accumulation over all layers at once — no loop
    ca_all    = torch.stack(cross_attn_store, dim=0)              # (n_layers, B, n_head, Td, Te)
    ca_cosmo  = ca_all[:, :, :, :, :N_COSMO_TOKENS]              # (n_layers, B, n_head, Td, 5)
    ca_patch  = ca_all[:, :, :, :, N_COSMO_TOKENS:]              # (n_layers, B, n_head, Td, N_patches)
    Td        = ca_all.shape[3]
    idx       = sort_idx_batch[None, :, None, None, :].expand(n_layers, B, n_heads, Td, N_patches)
    ca_patch_sorted = torch.gather(ca_patch, dim=4, index=idx)
    ca_reordered    = torch.cat([ca_cosmo, ca_patch_sorted], dim=4)  # (n_layers, B, n_head, Td, Te)
    valid_f         = (valid_mask.float()[None, :, None, :, None]) #.cpu()     # (1, B, 1, L_seq, 1)
    ca_masked       = ca_reordered * valid_f                          # (n_layers, B, n_head, Td, Te)
    cross_sum      += ca_masked.sum(dim=1).numpy()                    # (n_layers, n_head, Td, Te)

    # Count valid query positions (same across layers)
    cross_cnt += valid_mask.sum(dim=0).cpu().numpy()                 # (L_seq,)

print(f'\nDone. Computing averages...')

for h in hooks:
    h.remove()

# Average: divide by per-position count (L_seq,) → broadcast over layers, heads, encoder dims
cross_avg = cross_sum / np.maximum(cross_cnt[np.newaxis, np.newaxis, :, np.newaxis], 1)
cross_avg = cross_avg.astype(np.float32) # (n_layers, n_heads, L_seq, Te)

# ============================================================
# Plotting
# ============================================================
COSMO_TOKENS = {r'$\mathrm{\Omega_m}$', r'$\sigma_8$', r'$\mathrm{\Omega_b}$', r'$h$', r'$n_s$'}
_d_sample = [0, N_patches // 4, N_patches // 2, 3 * N_patches // 4, N_patches - 1]
x_ticks   = list(range(N_COSMO_TOKENS)) + [N_COSMO_TOKENS + d for d in _d_sample]
x_labels  = ([i for i in COSMO_TOKENS]
             + [f'D{d}' for d in _d_sample])


def plot_cross_avg_grid(attn_np, title, save_path):
    """attn_np : (n_heads, T_dec, Te)"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(title, fontsize=10)
    for hi in range(n_heads):
        row, col = hi // 4, hi % 4
        ax = axes[row, col]
        im = ax.imshow(
            attn_np[hi], cmap='viridis', aspect='auto',
            origin='upper', interpolation='nearest', vmin=0,
        )
        # Red line separating cosmo tokens from patch tokens
        ax.axvline(x=N_COSMO_TOKENS - 0.5, color='red', linestyle='-',
                   linewidth=1.2, alpha=0.9)
        ax.set_title(f'Head {hi + 1}', fontsize=9)
        ax.set_xlabel('Encoder token  [C0-C4 | density rank ↓]', fontsize=7)
        ax.set_ylabel('Decoder token position', fontsize=7)
        ax.set_xticks([t for t in x_ticks if t < Te])
        ax.set_xticklabels(
            [x_labels[i] for i, t in enumerate(x_ticks) if t < Te],
            fontsize=5, rotation=45,
        )
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')

'''
print('\n--- Averaged cross-attention figures ---')
for li in range(n_layers):
    # Trim rows that were never touched (beyond any sub-box's end_pos)
    valid_rows = int((cross_cnt > 0).sum())
    attn = cross_avg[li, :, :valid_rows, :]   # (n_heads, valid_rows, Te)
    print(f'  Layer {li + 1}: plotting {valid_rows} decoder token positions')
    plot_cross_avg_grid(
        attn,
        title=(
            f'Avg Cross-Attention — Layer {li + 1} — Sim {chosen_sim}'
            f'  ({N_subbox} sub-boxes)\n'
            f'x: [cosmo *{N_COSMO_TOKENS} | {N_patches} patches, density ↓]'
            f'  |  y: decoder token position'
        ),
        save_path=os.path.join(OUT_DIR, f'cross_avg_layer{li + 1:02d}.png'),
    )
'''

valid_rows = int((cross_cnt > 0).sum())


def _ax_setup(ax, im, title_str):
    ax.axvline(x=N_COSMO_TOKENS - 0.5, color='red', linestyle='-', linewidth=1.2, alpha=0.9)
    ax.set_title(title_str, fontsize=9)
    ax.set_xlabel('Encoder token  [C0-C4 | density rank ↓]', fontsize=7)
    ax.set_ylabel('Decoder token position', fontsize=7)
    ax.set_xticks([t for t in x_ticks if t < Te])
    ax.set_xticklabels(
        [x_labels[i] for i, t in enumerate(x_ticks) if t < Te],
        fontsize=5, rotation=45,
    )
    # y-ticks: mark x-position tokens (7 + 8*i) with halo index
    xpos_ticks  = [7 + 8 * i for i in range((valid_rows - 7 + 7) // 8 + 1) if 7 + 8 * i < valid_rows]
    xpos_labels = [f'#{i}' for i in range(len(xpos_ticks))]
    ax.set_yticks(xpos_ticks)
    ax.set_yticklabels(xpos_labels, fontsize=5, color='cyan')
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

'''
# Plot 1: head-averaged per layer — 2×4 grid (one panel per layer)
print('\n--- Head-averaged per layer (2×4) ---')
attn_head_avg = cross_avg[:, :, :valid_rows, :].mean(axis=1)   # (n_layers, valid_rows, Te)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(
    f'Avg Cross-Attention (head avg) — Sim {chosen_sim}  ({N_subbox} sub-boxes)\n'
    f'x: [cosmo ×{N_COSMO_TOKENS} | {N_patches} patches, density ↓]  |  y: decoder token position',
    fontsize=10,
)
for li in range(n_layers):
    row, col = li // 4, li % 4
    ax = axes[row, col]
    im = ax.imshow(attn_head_avg[li], cmap='viridis', aspect='auto',
                   origin='upper', interpolation='nearest', vmin=0)
    _ax_setup(ax, im, f'Layer {li + 1}')
plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'cross_avg_head_mean_per_layer.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {save_path}')
'''

# Plot 2: mean over both layers and heads — single panel
print('\n--- Layer+head averaged (single panel) ---')
attn_all_avg = cross_avg[:, :, :valid_rows, :].mean(axis=(0, 1))   # (valid_rows, Te)
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(
    f'Avg Cross-Attention (all layers & heads) — Sim {chosen_sim}  ({N_subbox} sub-boxes)\n'
    f'x: [cosmo ×{N_COSMO_TOKENS} | {N_patches} patches, density ↓]  |  y: decoder token position',
    fontsize=10,
)
im = ax.imshow(attn_all_avg, cmap='viridis', aspect='auto',
               origin='upper', interpolation='nearest', vmin=0)
_ax_setup(ax, im, f'All {n_layers} layers, all {n_heads} heads')
plt.tight_layout()
save_path = os.path.join(OUT_DIR, f'cross_avg_all_mean_{chosen_sim}.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {save_path}')

print(f'\nAll figures saved to: {OUT_DIR}')
