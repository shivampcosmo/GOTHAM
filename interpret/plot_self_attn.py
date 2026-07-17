import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

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
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': False,
    'ytick.right': False,
})

# ============================================================
# Config — edit here
# ============================================================
OUT_DIR        = '/u/yzhang116/NN/figures/visualize/results/'
OUT_NPY_DIR    = '/work/hdd/bdne/yzhang116/cp_visualize/during_gen'

N_COSMO_TOKENS = 6
N_HEADER       = 7   # 5 cosmo tokens + N_halo
chosen_sim     = 66
plot_range     = N_HEADER + 5 * 8

COSMO_TOKENS  = ['<START>',r'$\mathrm{\Omega_m}$', r'$\sigma_8$', r'$\mathrm{\Omega_b}$', r'$h$', r'$n_s$']

self_avg = np.load(os.path.join(OUT_NPY_DIR, f'self_attn_per_layer_head_cosmo_{chosen_sim}.npy'))  # (n_layers, n_heads, T, T)
# include the leading <START> token plus the meaningful tokens
self_avg = self_avg[:, :, :plot_range, :plot_range]
n_layers, n_heads, valid_rows, _ = self_avg.shape

# mean over both layers and heads
attn_all_avg = self_avg.mean(axis=(0, 1))   # (valid_rows, valid_rows)

# MNRAS single-column width ~3.4 in
fig, ax = plt.subplots(figsize=(3.4, 3.4))
im = ax.imshow(attn_all_avg, cmap='Blues', aspect='auto',
               origin='upper', interpolation='nearest',
               norm=LogNorm(vmin=0.01, vmax=0.2))
ax.set_box_aspect(1)
ax.autoscale(False)


def _hline_tri(ax, y, **kwargs):
    """Horizontal line at row y, kept only in the lower-left triangle (x <= y)."""
    ax.plot([-0.5, y], [y, y], **kwargs)


def _vline_tri(ax, x, **kwargs):
    """Vertical line at column x, kept only in the lower-left triangle (y >= x)."""
    ax.plot([x, x], [x, valid_rows - 0.5], **kwargs)


# header separators: cosmo tokens | N_halo | halo blocks
#_vline_tri(ax, N_COSMO_TOKENS - 0.5, color='grey', linewidth=0.8, zorder=5)
#_hline_tri(ax, N_COSMO_TOKENS - 0.5, color='grey', linewidth=0.8, zorder=5)
_vline_tri(ax, N_HEADER - 0.5, color='black', linewidth=0.8, zorder=5)
_hline_tri(ax, N_HEADER - 0.5, color='black', linewidth=0.8, zorder=5)

ax.set_xticks([])
ax.set_yticks([])

assert (valid_rows - N_HEADER) % 8 == 0, f'unexpected valid_rows={valid_rows}'
n_halo = (valid_rows - N_HEADER) // 8

XYZ_COLOR = 'tab:blue'
V_COLOR   = 'tab:blue'

# --- y-axis labels (left side) ---
ytrans = ax.get_yaxis_transform()  # x in axes fraction, y in data coords
for j, label in enumerate(COSMO_TOKENS):
    ax.text(-0.01, j, label, transform=ytrans, ha='right', va='center', fontsize=4)
ax.text(-0.01, N_COSMO_TOKENS, r'$N_\mathrm{halo}$', transform=ytrans, ha='right', va='center', fontsize=4)

# --- x-axis labels (bottom); same labels as the y-axis, rotated to fit ---
xtrans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
for j, label in enumerate(COSMO_TOKENS):
    ax.text(j, -0.01, label, transform=xtrans, ha='center', va='top',
            fontsize=4, rotation=90, clip_on=False)
ax.text(N_COSMO_TOKENS, -0.01, r'$N_\mathrm{halo}$', transform=xtrans, ha='center', va='top',
        fontsize=5, rotation=90, clip_on=False)

for i in range(n_halo):
    row0 = N_HEADER + 8 * i

    # per-row-group labels, aligned with their corresponding rows/columns;
    # the same labels are used on both axes since this is self-attention
    ax.text(-0.01, row0 + 1, f'$x_{{{i + 1}}}, y_{{{i + 1}}}, z_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=5)
    ax.text(-0.01, row0 + 3, f'$M_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=5)
    ax.text(-0.01, row0 + 5, f'$v_{{x,{i + 1}}}, v_{{y,{i + 1}}}, v_{{z,{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=5)
    ax.text(-0.01, row0 + 7, f'$c_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=5)

    ax.text(row0 + 1, -0.01, f'$x_{{{i + 1}}}, y_{{{i + 1}}}, z_{{{i + 1}}}$',
            transform=xtrans, ha='center', va='top', fontsize=5, rotation=90, clip_on=False)
    ax.text(row0 + 3, -0.01, f'$M_{{{i + 1}}}$',
            transform=xtrans, ha='center', va='top', fontsize=5, rotation=90, clip_on=False)
    ax.text(row0 + 5, -0.01, f'$v_{{x,{i + 1}}}, v_{{y,{i + 1}}}, v_{{z,{i + 1}}}$',
            transform=xtrans, ha='center', va='top', fontsize=5, rotation=90, clip_on=False)
    ax.text(row0 + 7, -0.01, f'$c_{{{i + 1}}}$',
            transform=xtrans, ha='center', va='top', fontsize=5, rotation=90, clip_on=False)

    # dashed lines bracketing (x, y, z) — lower-left triangle only
    _hline_tri(ax, row0 - 0.5, color=XYZ_COLOR, linestyle='--', linewidth=0.6, zorder=5)
    #_hline_tri(ax, row0 + 2.5, color=XYZ_COLOR, linestyle='--', linewidth=0.6, zorder=5)
    _vline_tri(ax, row0 - 0.5, color=XYZ_COLOR, linestyle='--', linewidth=0.6, zorder=5)
    #_vline_tri(ax, row0 + 2.5, color=XYZ_COLOR, linestyle='--', linewidth=0.6, zorder=5)

    # dotted lines bracketing (vx, vy, vz) — lower-left triangle only
    _hline_tri(ax, row0 + 3.5, color=V_COLOR, linestyle=':', linewidth=0.6, zorder=5)
    #_hline_tri(ax, row0 + 6.5, color=V_COLOR, linestyle=':', linewidth=0.6, zorder=5)
    _vline_tri(ax, row0 + 3.5, color=V_COLOR, linestyle=':', linewidth=0.6, zorder=5)
    #_vline_tri(ax, row0 + 6.5, color=V_COLOR, linestyle=':', linewidth=0.6, zorder=5)

    # orange dots: row = c_i, columns = (x,y,z)_{i'} for every earlier halo i' < i
    c_row = row0 + 7
    for j in range(i+1):
        prev_row0 = N_HEADER + 8 * j
        ax.plot(prev_row0, c_row, 'o', color='orange', markersize=0.8, zorder=6)

    # green dot: row = (vx,vy,vz)_i, column = M_{i-1}
    if i >= 1:
        prev_row0 = N_HEADER + 8 * (i - 1)
        ax.plot(prev_row0 + 3, row0 + 2, 'o', color='red', markersize=0.8, zorder=6)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Attention weight', fontsize=6)
cbar.ax.tick_params(labelsize=5)
cbar.set_ticks(np.append(np.arange(0.01, 0.101, 0.01), 0.2))
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
cbar.ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: ''))

fig.tight_layout()
save_path = os.path.join(OUT_DIR, f'self_avg_all_cosmo_{chosen_sim}')
plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {save_path}.pdf / .png')
