"""save_embeddings_logits.py

Save model embeddings and logits for SIMID=66.

Saves to OUT_DIR/sim{SIM_ID}_embeddings_logits.npz:
  wte         : (vocab_size, n_embd)     — token embedding table
  pos_embed   : (1, 512, embed_dim)      — ViT positional embedding
  logits_100  : (100, L-1, vocab_size)   — decoder logits for sub-boxes 0-99
  sentence    : (100, L)                 — raw token IDs for sub-boxes 0-99

Usage:
    cd /u/yzhang116/NN/quijote
    python save_embed_logits.py
"""

import sys, os
sys.path.insert(0, '/u/yzhang116/NN/quijote')

import torch
import numpy as np
import pickle as pk
from model_enc_dec_cos import HaloDecoderModel

# ============================================================
# Config — edit here
# ============================================================
CHECKPOINT     = '/work/hdd/bdne/yzhang116/checkpoints_quijote/1872236/checkpoint_cos_cross_entropy_epoch_0_step_7050_embed_384_batch_800_lrmax_1e-05_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt'
DMO_DIR        = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3'
HALO_DIR       = '/work/hdd/bdne/yzhang116/halo_sentence_full_quijote'
META_PATH  = '/work/nvme/bdne/yzhang116/quijote_halos/sentence_params.pkl'
OUT_DIR    = '/work/hdd/bdne/yzhang116/cp_visualize/'
SIM_ID     = 66
N_SUBBOX   = 100   # sub-boxes to process (0..N_SUBBOX-1)
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
wte       = model.transformer.wte.weight.detach().cpu()   # (vocab_size, n_embd)
pos_embed = model.cnn3D.pos_embed.detach().cpu()          # (1, 512, embed_dim)
print(f'wte shape:       {tuple(wte.shape)}')
print(f'pos_embed shape: {tuple(pos_embed.shape)}')

# ---- Load SIMID=66 data ----
chosen_sim  = np.random.choice(sim_numbers) if SIM_ID is None else SIM_ID
dm_path     = os.path.join(DMO_DIR,  f'{chosen_sim}/dmo_fields_subvols_grid_8_LH_{chosen_sim}.npy')
halo_path   = os.path.join(HALO_DIR, f'halo_sentence_LH_{chosen_sim}.npy')
print(f'Loading sim {SIM_ID}: {dm_path}')

DM_all  = np.load(dm_path)   # (N_total_subbox, D, H, W, C)
gal_temp  = np.load(halo_path) # (N_total_subbox, L)
N_subbox = DM_all.shape[0]
end_token_index = np.argmax(gal_temp == end_token, axis=1)
n_halos = (end_token_index-6) // 8
gal_all = np.zeros((N_subbox, gal_temp.shape[1] + 1), dtype=gal_temp.dtype)
gal_all[:,:6] = gal_temp[:, :6]
gal_all[:,6] = n_halos
gal_all[:,7:] = gal_temp[:, 6:]
assert DM_all.shape[0] >= N_SUBBOX, \
    f'Only {DM_all.shape[0]} sub-boxes available, requested {N_SUBBOX}'

sentence = gal_all[:N_SUBBOX]  # (100, L)
print(f'sentence shape: {tuple(sentence.shape)}')

# ---- lm_head hook to capture logits ----
logits_store = []

def lm_head_hook(module, args, output):
    logits_store.append(output.detach().cpu().float())   # (B, T, vocab_size)

hook = model.lm_head.register_forward_hook(lm_head_hook)

# ---- Batched forward passes over sub-boxes 0..N_SUBBOX-1 ----
batch_start = 0
batch_end = N_SUBBOX
B = batch_end - batch_start
print(f'  sub-boxes {batch_start}–{batch_end - 1}', end='\r', flush=True)

DM  = torch.tensor(DM_all[batch_start:batch_end], dtype=torch.float32, device=DEVICE)
DM  = DM.moveaxis(-1, 1)                                           # (B, C, D, H, W)
gal = torch.tensor(gal_all[batch_start:batch_end], dtype=torch.long, device=DEVICE)

X = gal[:, :-1]           # (B, L-1)
print(f'Input DM shape: {tuple(DM.shape)}')
print(f'Input gal shape: {tuple(gal.shape)}')
Y = gal[:, 1:].clone()    # (B, L-1)
Y[:, :5] = pad_token
pad_mask = torch.logical_not(X != pad_token)
masked_logits_t = torch.zeros(pad_mask.shape, device=DEVICE, dtype=torch.float32)
MASK = masked_logits_t.masked_fill(pad_mask, float('-inf'))[:, None, :]  # (B, 1, L-1)

with torch.no_grad():
    _ = model(X, DM, maskd=MASK, targets=Y)

hook.remove()
print(f'\nForward passes done. Concatenating logits...')

# ---- Concatenate all batches: (100, L-1, vocab_size) ----
logits_all = torch.cat(logits_store, dim=0)   # (100, L-1, vocab_size)
print(f'logits_all shape: {tuple(logits_all.shape)}')

# ---- Save ----
out_path = os.path.join(OUT_DIR, f'sim{SIM_ID}_embeddings_logits.npz')
np.savez(
    out_path,
    wte=wte.numpy(),               # (vocab_size, n_embd)
    pos_embed=pos_embed.numpy(),   # (1, 512, embed_dim)
    logits_100=logits_all.numpy(), # (100, L-1, vocab_size)
    sentence=sentence,             # (100, L)
)
print(f'Saved: {out_path}')
print({k: np.load(out_path)[k].shape for k in np.load(out_path)})