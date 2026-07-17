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
OUT_NPY_DIR    = '/work/hdd/bdne/yzhang116/cp_visualize/during_gen/'

N_COSMO_TOKENS = 5
chosen_sim = 66

COSMO_TOKENS  = [r'$\mathrm{\Omega_m}$', r'$\sigma_8$', r'$\mathrm{\Omega_b}$', r'$h$', r'$n_s$']
cosmo_labels = r'$\mathrm{\Omega_m}$' + r',' + r'$\sigma_8$' + r',' + r'$\mathrm{\Omega_b}$' + r',' + r'$h$' + r',' + r'$n_s$'

cross_avg = np.load(os.path.join(OUT_NPY_DIR, f'cross_attn_per_layer_head_cosmo_{chosen_sim}_5halos.npy'))  # (n_layers, n_heads, valid_rows, Te)
cross_avg = cross_avg[:, :, :, :]
n_layers, n_heads, valid_rows, Te = cross_avg.shape

# mean over both layers and heads
attn_all_avg = cross_avg.mean(axis=(0, 1))   # (valid_rows, Te)

# MNRAS single-column width ~3.4 in
fig, ax = plt.subplots(figsize=(3.4, 3.0))
im = ax.imshow(attn_all_avg, cmap='Blues', aspect='auto',
               origin='upper', interpolation='nearest',
               norm=LogNorm(vmin=0.01, vmax=0.1))
ax.set_box_aspect(1)

ax.axvline(x=N_COSMO_TOKENS - 0.5, color='black', linewidth=0.8, alpha=1.)

# x-axis: no ticks; cosmological parameter tokens are written in a row
# below the axis, approximately above their column position; remaining
# columns are DM-field tokens, marked instead with an arrow
ax.set_xticks([])

trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
center = (N_COSMO_TOKENS - 1) / 2
ax.text(center, -0.01, cosmo_labels, transform=trans, ha='center', va='top',
            fontsize=5, clip_on=False)
'''
spread = 3.0
for j, label in enumerate(COSMO_TOKENS):
    xpos = center + (j - center) * spread
    ax.text(xpos, -0.01, label, transform=trans, ha='center', va='top',
            fontsize=6, clip_on=False)
'''
ax.annotate('', xy=(Te - 0.5, -0.06), xytext=(N_COSMO_TOKENS - 0.5, -0.06),
             xycoords=trans, textcoords=trans,
             arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
             annotation_clip=False)
ax.text((N_COSMO_TOKENS - 0.5 + Te - 0.5) / 2, -0.08,
        'DM fields (density decreasing)',
        transform=trans, ha='center', va='top', fontsize=6, clip_on=False)

# y-axis: no ticks; instead use solid separators between halos, dashed
# lines bracketing the (x, y, z) and (vx, vy, vz) sub-groups, and
# row-group labels on the left
assert (valid_rows - 7) % 8 == 0, f'unexpected valid_rows={valid_rows}'
n_halo = (valid_rows - 7) // 8
ax.set_yticks([])

XYZ_COLOR = 'tab:blue'
V_COLOR   = 'tab:blue'
ytrans = ax.get_yaxis_transform()  # x in axes fraction, y in data coords

ax.text(-0.01, 0, '<START>', transform=ytrans, ha='right', va='center', fontsize=4)
ax.text(-0.01, 1, r'$\mathrm{\Omega_m}$', transform=ytrans, ha='right', va='center', fontsize=4)
ax.text(-0.01, 2, r'$\sigma_8$', transform=ytrans, ha='right', va='center', fontsize=4)
ax.text(-0.01, 3, r'$\mathrm{\Omega_b}$', transform=ytrans, ha='right', va='center', fontsize=4)
ax.text(-0.01, 4, r'$h$', transform=ytrans, ha='right', va='center', fontsize=4)
ax.text(-0.01, 5, r'$n_s$', transform=ytrans, ha='right', va='center', fontsize=4)
ax.text(-0.01, 6, r'$N_\mathrm{halo}$', transform=ytrans, ha='right', va='center', fontsize=4)

for i in range(n_halo):
    row0 = 7 + 8 * i

    # solid line separating this halo's block from the previous one
    #ax.axhline(y=row0 - 0.5, color='black', linewidth=0.6)

    # per-row-group labels, aligned with their corresponding rows
    ax.text(-0.01, row0 + 1, f'$x_{{{i + 1}}}, y_{{{i + 1}}}, z_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=5)
    ax.text(-0.01, row0 + 3, f'$M_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=5)
    ax.text(-0.01, row0 + 5, f'$v_{{x,{i + 1}}}, v_{{y,{i + 1}}}, v_{{z,{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=5)
    ax.text(-0.01, row0 + 7, f'$c_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=5)

    # dashed lines bracketing (x, y, z)
    ax.axhline(y=row0 - 0.5, color=XYZ_COLOR, linestyle='--', linewidth=0.6, zorder=5)
    #ax.axhline(y=row0 + 2.5, color=XYZ_COLOR, linestyle='--', linewidth=0.6, zorder=5)

    # dashed lines bracketing (vx, vy, vz)
    ax.axhline(y=row0 + 3.5, color=V_COLOR, linestyle=':', linewidth=0.6, zorder=5)
    #ax.axhline(y=row0 + 6.5, color=V_COLOR, linestyle=':', linewidth=0.6, zorder=5)

# bottom border of the last halo block
#ax.axhline(y=valid_rows - 0.5, color='black', linewidth=0.6)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Attention weight', fontsize=5)
cbar.ax.tick_params(labelsize=5)
cbar.set_ticks(np.arange(0.01, 0.101, 0.01))
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
cbar.ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: ''))

fig.tight_layout()
save_path = os.path.join(OUT_DIR, f'cross_avg_all_cosmo_{chosen_sim}')
plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {save_path}.pdf / .png')
