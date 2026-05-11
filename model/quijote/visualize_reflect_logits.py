"""visualize_logits_reynolds.py

Capture and plot the logits produced during reflect-Reynolds generation.
Processes subboxes in batches of BATCH_SIZE; only saves plots for subboxes
where the true catalog has n_halos > 2.
"""

import os
import sys
sys.path.insert(0, '/u/yzhang116/NN/quijote')
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
from model_enc_dec_reflect import HaloDecoderModel

# ------------------------------------------------------------------ config --
CKPT = (
    '/work/hdd/bdne/yzhang116/checkpoints_quijote/1872236/'
    'checkpoint_cos_cross_entropy_epoch_0_step_7050_embed_384_batch_800_'
    'lrmax_1e-05_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_'
    'patch_4_dropout_0.0.pt'
)
DMO_DIR    = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3'
PARAM_FILE = '/work/nvme/bdne/yzhang116/quijote_halos/quijote_params.txt'
SIMID      = 66
N_HALOS    = 4          # rows in the figure
NPROPS     = 8          # columns (x, y, z, mass, vx, vy, vz, conc)
TEMPERATURE = 1.0
SEED       = 42
BATCH_SIZE = 100
OUT_DIR    = f'logits_vis_reynolds_sim{SIMID}'
# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.preferred_blas_library("cublaslt")

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

nvocab      = 131
start_token = nvocab + 1   # 132
pad_token   = nvocab + 3   # 134
end_token   = nvocab + 4   # 135

# ---- load model --------------------------------------------------------
print('Loading checkpoint …', flush=True)
ckpt = torch.load(CKPT, map_location=dev)
sd   = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
cfg  = ckpt['config']

model = HaloDecoderModel(cfg).to(dev)
model.load_state_dict(sd, strict=False)
model.eval()
print('Model ready.', flush=True)

# ---- load data ---------------------------------------------------------
all_param = np.loadtxt(PARAM_FILE)
dmo_all   = np.load(f'{DMO_DIR}/{SIMID}/dmo_fields_subvols_grid_8_LH_{SIMID}.npy')
dmo_all   = torch.from_numpy(dmo_all).bfloat16()   # (nsubox, D, H, W, C) on CPU
dmo_all   = torch.moveaxis(dmo_all, -1, 1)          # (nsubox, C, D, H, W)

param = all_param[SIMID]
param = np.repeat(param[None, :], dmo_all.shape[0], axis=0)
params_all = torch.tensor(param).bfloat16()         # (nsubox, nparam) on CPU

# ---- load true halo sentences ------------------------------------------
TRUE_SENTENCE_FILE = (
    f'/work/hdd/bdne/yzhang116/halo_sentence_full_quijote/'
    f'halo_sentence_LH_{SIMID}.npy'
)
true_sentence_all = np.load(TRUE_SENTENCE_FILE)     # (nsubox, L)
print(f'True sentences shape: {true_sentence_all.shape}', flush=True)

# ---- filter subboxes where true n_halos > 2 ----------------------------
def get_nhalos(sentence):
    starts = np.where(sentence == start_token)[0]
    ends   = np.where(sentence == end_token)[0]
    if len(starts) == 0 or len(ends) == 0:
        return 0
    return int((ends[0] - starts[0] - 1) / NPROPS)

valid_subboxes = [i for i in range(true_sentence_all.shape[0])
                  if get_nhalos(true_sentence_all[i]) > 2]
print(f'Subboxes with n_halos > 2: {len(valid_subboxes)}', flush=True)

# ---- helpers -----------------------------------------------------------
REFLECT_AXIS = [None, 0, 1, 2]
P = 40    # position token range 0..P-1
V = 131   # max velocity token index (inclusive)

def unreflect(lg, prop_idx, config_idx):
    axis = REFLECT_AXIS[config_idx]
    if axis is None:
        return lg
    out = lg.copy()
    if prop_idx == axis:
        out[:P] = lg[:P][::-1]
    elif prop_idx == axis + 4:
        out[:V + 1] = lg[:V + 1][::-1]
    return out

def softmax(x):
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()

PROP_NAMES = ['x', 'y', 'z', 'mass', 'vx', 'vy', 'vz', 'conc']
CFG_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
CFG_LABELS = ['cfg0 (identity)', 'cfg1 (x-flip)', 'cfg2 (y-flip)', 'cfg3 (z-flip)']

# ---- hook (defined once; captures all batch items) ---------------------
# raw_captures: list of (B, vocab_size) arrays, one entry per lm_head call
raw_captures = []

def _hook(module, inp, out):
    raw_captures.append(out[:, -1, :].float().detach().cpu().numpy())

# ---- plot helper -------------------------------------------------------
def plot_subbox(subbox_idx, tokens, true_tokens, b, VS):
    """Build and save one figure for subbox_idx (index b in current batch)."""
    captured = {}
    for hi in range(N_HALOS):
        for pi in range(NPROPS):
            base = 4 + (hi * NPROPS + pi) * 4   # skip N_halo block (4 calls)
            if base + 3 >= len(raw_captures):
                break
            logits_configs = []
            for ci in range(4):
                raw = raw_captures[base + ci][b] / TEMPERATURE
                lg  = unreflect(raw, pi, ci)
                logits_configs.append(lg)
            avg_lg = np.mean(logits_configs, axis=0)
            captured[(hi, pi)] = {
                'probs'   : [softmax(lg) for lg in logits_configs],
                'avg_prob': softmax(avg_lg),
            }

    token_axis = np.arange(VS)
    fig, axes = plt.subplots(N_HALOS, NPROPS,
                             figsize=(2.8 * NPROPS, 2.5 * N_HALOS),
                             squeeze=False)
    fig.suptitle(
        f'Softmax probs — sim {SIMID}, subbox {subbox_idx}  |  '
        f'dotted=sampled  green=truth',
        fontsize=10,
    )

    for hi in range(N_HALOS):
        for pi in range(NPROPS):
            ax  = axes[hi][pi]
            key = (hi, pi)
            if key not in captured:
                ax.set_visible(False)
                continue

            data = captured[key]

            for ci in range(4):
                ax.plot(token_axis, data['probs'][ci],
                        color=CFG_COLORS[ci], lw=0.8, alpha=0.75,
                        label=CFG_LABELS[ci] if (hi == 0 and pi == 0) else None)
            ax.plot(token_axis, data['avg_prob'],
                    color='black', lw=1.5, ls='--',
                    label='average' if (hi == 0 and pi == 0) else None)

            tok_offset = 7 + hi * NPROPS + pi
            if tok_offset < len(tokens):
                sampled = int(tokens[tok_offset])
                ax.axvline(sampled, color='k', lw=0.9, ls=':', alpha=0.6)
                is_end = (pi == 0 and sampled == end_token)
                title_str = f'{PROP_NAMES[pi]}  (tok={sampled})'
                if is_end:
                    title_str += '  [END TOKEN]'
                ax.set_title(title_str, fontsize=7.5, color='red' if is_end else 'black')
            else:
                ax.set_title(PROP_NAMES[pi], fontsize=7.5)

            if tok_offset < len(true_tokens):
                true_tok = int(true_tokens[tok_offset - 1])
                ax.axvline(true_tok, color='limegreen', lw=0.9, ls='--', alpha=0.8,
                           label='truth' if (hi == 0 and pi == 0) else None)

            if pi == 0:
                ax.set_ylabel(f'halo {hi}', fontsize=8)

            xlim = (0, P - 1) if pi < 3 else (0, VS - 1)
            ax.set_xlim(*xlim)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=5)
            ax.set_xlabel('token index', fontsize=5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = os.path.join(OUT_DIR, f'logits_vis_reynolds_{subbox_idx}.pdf')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out_path}', flush=True)

# ---- main batch loop ---------------------------------------------------
n_batches = (len(valid_subboxes) + BATCH_SIZE - 1) // BATCH_SIZE
h = model.lm_head.register_forward_hook(_hook)

for batch_i in range(n_batches):
    batch_indices = valid_subboxes[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
    B = len(batch_indices)
    print(f'Batch {batch_i+1}/{n_batches} — {B} subboxes '
          f'({batch_indices[0]}…{batch_indices[-1]})', flush=True)

    dmo_batch    = dmo_all[batch_indices].to(dev)
    params_batch = params_all[batch_indices].to(dev)

    raw_captures.clear()
    torch.manual_seed(SEED)
    if dev.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)

    print('  Running generation …', flush=True)
    with torch.no_grad():
        with ctx:
            output_idx = model.generate_with_reflection_reynolds(
                dmo_batch,
                params=params_batch,
                max_new_tokens=289,
                start_token=start_token,
                end_token=end_token,
                pad_token=pad_token,
                temperature=TEMPERATURE,
            )
    print(f'  lm_head calls captured: {len(raw_captures)}', flush=True)

    VS = raw_captures[0].shape[1]

    for b, subbox_idx in enumerate(batch_indices):
        tokens      = output_idx[b].cpu().numpy()
        true_tokens = true_sentence_all[subbox_idx]
        plot_subbox(subbox_idx, tokens, true_tokens, b, VS)

    del dmo_batch, params_batch, output_idx

h.remove()
print('All done.', flush=True)
