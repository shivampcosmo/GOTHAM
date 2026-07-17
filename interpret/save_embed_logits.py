"""save_embeddings_logits.py

Save model embeddings and logits for SIMID=66.

Saves to OUT_DIR/sim{SIM_ID}_embeddings_logits.npz:
  wte            : (vocab_size, n_embd)     — token embedding table
  pos_embed      : (1, 512, embed_dim)      — ViT positional embedding
  whe            : (max_nhalo, n_embd)      — halo-index embedding table
  wprope         : (nprops, n_embd)         — property-index embedding table
  x_pre          : (100, L-1, n_embd)       — x before any decoder block
  x_block_{k}    : (100, L-1, n_embd)       — x after decoder block k (k=0..7)
  logits_100     : (100, L-1, vocab_size)   — decoder logits for sub-boxes 0-99
  x_lnf          : (100, L-1, n_embd)       — x after ln_f (before lm_head)
  sentence       : (100, L)                 — raw token IDs for sub-boxes 0-99

Usage:
    cd /u/yzhang116/NN/quijote
    python save_embeddings_logits.py
"""

import sys, os
sys.path.insert(0, '/u/yzhang116/NN/quijote')

import torch
import numpy as np
import pickle as pk
from model_enc_dec_cos_fast import HaloDecoderModel

#ckpt = 2453077
#ckpt = 2453036
#step = 111000

# ============================================================
# Config — edit here
# ============================================================
CHECKPOINT     = '/work/hdd/bdne/yzhang116/checkpoints_quijote/1872236/checkpoint_cos_cross_entropy_epoch_0_step_7050_embed_384_batch_800_lrmax_1e-05_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt'
#CHECKPOINT     = f'/work/hdd/bdne/yzhang116/checkpoints_quijote/{ckpt}/checkpoint_5patch_cross_entropy_epoch_2_step_{step}_embed_384_batch_800_lrmax_0.0001_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_5_dropout_0.0.pt'
HALO_DIR       = '/work/hdd/bdne/yzhang116/cp_visualize/during_gen'
META_PATH  = '/work/nvme/bdne/yzhang116/quijote_halos/sentence_params.pkl'
OUT_DIR        = '/work/hdd/bdne/yzhang116/cp_visualize/during_gen'
SIM_ID     = 66
DMO_DIR        = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3/'
N_SUBBOX   = 1000   # sub-boxes to process (0..N_SUBBOX-1)
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
HaloConfig['flash']  = True
model = HaloDecoderModel(HaloConfig).to(DEVICE)
sd    = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
model.load_state_dict(sd, strict=False)
model.eval()
print(f'Loaded checkpoint: {CHECKPOINT}')

# ---- Static embeddings (no forward pass needed) ----
wte       = model.transformer.wte.weight.detach().cpu()    # (vocab_size, n_embd)
pos_embed = model.cnn3D.pos_embed.detach().cpu()           # (1, 512, embed_dim)
whe       = model.transformer.whe.weight.detach().cpu()    # (max_nhalo, n_embd)
wprope    = model.transformer.wprope.weight.detach().cpu() # (nprops, n_embd)
print(f'wte shape:       {tuple(wte.shape)}')
print(wte[:2,:10])
print(f'pos_embed shape: {tuple(pos_embed.shape)}')
print(f'whe shape:       {tuple(whe.shape)}')
print(f'whe[:5,:10]:\n{whe[:5,:10]}')
print(f'wprope shape:    {tuple(wprope.shape)}')

# ---- Load SIMID=66 data ----
chosen_sim  = np.random.choice(sim_numbers) if SIM_ID is None else SIM_ID
dm_path     = os.path.join(DMO_DIR,  f'{chosen_sim}/dmo_fields_subvols_grid_8_LH_{chosen_sim}.npy')
halo_path   = os.path.join(HALO_DIR, f'generated_halo_sentence_{chosen_sim}.npy')
print(f'Loading sim {SIM_ID}: {dm_path}')


DM_all  = np.load(dm_path)   # (N_total_subbox, D, H, W, C)
gal_all  = np.load(halo_path) # (N_total_subbox, L)
N_subbox = DM_all.shape[0]
has_end = np.any(gal_all == end_token, axis=1)
end_token_index = np.argmax(gal_all == end_token, axis=1)
assert DM_all.shape[0] >= N_SUBBOX, \
    f'Only {DM_all.shape[0]} sub-boxes available, requested {N_SUBBOX}'

select_index = np.where(has_end & ((end_token_index - 7) % 8 == 0))[0]
DM_all = DM_all[select_index]
gal_all = gal_all[select_index]
# Generated sequences can keep sampling end_token as filler after the real end
# (instead of pad_token); keep only the first end_token per row and scrub
# everything after it to pad_token.
end_token_index_sel = end_token_index[select_index]
after_end = np.arange(gal_all.shape[1])[None, :] > end_token_index_sel[:, None]
gal_all[after_end] = pad_token

sentence = gal_all[:N_SUBBOX]  # (100, L)
print(f'sentence shape: {tuple(sentence.shape)}')

# ---- Hooks to capture intermediate x tensors and logits ----
logits_store  = []
x_pre_store   = []                          # x before any block
x_block_store = [[] for _ in range(8)]      # x after blocks 0-7
x_lnf_store   = [] 

def lm_head_hook(module, args, output):
    logits_store.append(output.detach().cpu().float())   # (B, T, vocab_size)

def make_x_hook(store):
    def hook(module, args, output):
        store.append(output.detach().cpu().float())      # (B, T, n_embd)
    return hook

hook_lm   = model.lm_head.register_forward_hook(lm_head_hook)
hook_lnf  = model.transformer.ln_f.register_forward_hook(make_x_hook(x_lnf_store))
hook_pre  = model.transformer.drop.register_forward_hook(make_x_hook(x_pre_store))
hooks_blk = [
    model.transformer.h[k].register_forward_hook(make_x_hook(x_block_store[k]))
    for k in range(8)
]

# ---- Batched forward passes over sub-boxes 0..N_SUBBOX-1 ----
batch_start = 0
batch_end = N_SUBBOX
B = batch_end - batch_start
print(f'  sub-boxes {batch_start}–{batch_end - 1}', end='\r', flush=True)

DM  = torch.tensor(DM_all[batch_start:batch_end], dtype=torch.float32, device=DEVICE)
DM  = DM.moveaxis(-1, 1)                                           # (B, C, D, H, W)
gal = torch.tensor(gal_all[batch_start:batch_end], dtype=torch.long, device=DEVICE)

X = gal[:, :-1]           # (B, L-1)
Y = gal[:, 1:].clone()    # (B, L-1)
Y[:, :5] = pad_token
pad_mask = torch.logical_not(X != pad_token)
masked_logits_t = torch.zeros(pad_mask.shape, device=DEVICE, dtype=torch.float32)
MASK = masked_logits_t.masked_fill(pad_mask, float('-inf'))[:, None, :]  # (B, 1, L-1)

with torch.no_grad():
    _ = model(X, DM, maskd=MASK, targets=Y)

hook_lm.remove()
hook_pre.remove()
hook_lnf.remove()
for h in hooks_blk:
    h.remove()
print(f'\nForward passes done. Concatenating tensors...')

# ---- Concatenate all batches ----
logits_all  = torch.cat(logits_store,  dim=0)   # (100, L-1, vocab_size)
x_pre_all   = torch.cat(x_pre_store,   dim=0)   # (100, L-1, n_embd)
x_block_all = [torch.cat(x_block_store[k], dim=0) for k in range(8)]  # each (100, L-1, n_embd)
x_lnf_all   = torch.cat(x_lnf_store, dim=0)                           # (100, L-1, n_embd)
print(f'logits_all shape:    {tuple(logits_all.shape)}')
print(f'x_pre shape:         {tuple(x_pre_all.shape)}')
for k in range(8):
    print(f'x_block_{k} shape:    {tuple(x_block_all[k].shape)}')
print(f'x_lnf shape:         {tuple(x_lnf_all.shape)}')

# ---- Save ----
out_path = os.path.join(OUT_DIR, f'sim{SIM_ID}_embeddings_logits_blocks.npz')
save_dict = dict(
    wte=wte.numpy(),               # (vocab_size, n_embd)
    pos_embed=pos_embed.numpy(),   # (1, 512, embed_dim)
    whe=whe.numpy(),               # (max_nhalo, n_embd)
    wprope=wprope.numpy(),         # (nprops, n_embd)
    x_pre=x_pre_all.numpy(),       # (100, L-1, n_embd)
    x_lnf=x_lnf_all.numpy(),      # (100, L-1, n_embd)
    logits_100=logits_all.numpy(), # (100, L-1, vocab_size)
    sentence=sentence,             # (100, L)
)
for k in range(8):
    save_dict[f'x_block_{k}'] = x_block_all[k].numpy()  # (100, L-1, n_embd)
np.savez(out_path, **save_dict)
print(f'Saved: {out_path}')
print({k: np.load(out_path)[k].shape for k in np.load(out_path)})
