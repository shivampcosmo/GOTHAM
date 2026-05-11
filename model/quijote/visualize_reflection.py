"""
Visualize the 4 reflection configurations of a single sub-box.

Configurations:
  0 - identity
  1 - reflect x  (flip spatial dim 0, negate channels 7 & 15)
  2 - reflect y  (flip spatial dim 1, negate channels 8 & 16)
  3 - reflect z  (flip spatial dim 2, negate channels 9 & 17)

Each figure has 18 panels (one per channel), showing the middle z-slice.
The same panel uses the same color scale across all 4 figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_PATH = '/work/hdd/bdne/spandey3/quijote_LH_discodj/full_rhog_LH_np_512_nsnap_3/66/dmo_fields_subvols_grid_8_LH_66.npy'
OUT_DIR   = '/u/yzhang116/NN/quijote/figures_reflection'
os.makedirs(OUT_DIR, exist_ok=True)

SUBBOX_IDX = 0   # which sub-box to visualize
SLICE_IDX  = 4   # which z-slice (0..7); middle = 4

# ------------------------------------------------------------------
# Channel metadata (18 channels = snap0 + snap1 + snap2)
# snap0: ch 0-1   (dens, log_dens)
# snap1: ch 2-9   (dens, log_dens, pad1, log_pad1, pad2, vx, vy, vz)
# snap2: ch 10-17 (same as snap1)
# ------------------------------------------------------------------
CHANNEL_NAMES = [
    'snap0 dens',    'snap0 log_dens',
    'snap1 dens',    'snap1 log_dens', 'snap1 pad1', 'snap1 log_pad1', 'snap1 pad2',
    'snap1 vx',      'snap1 vy',       'snap1 vz',
    'snap2 dens',    'snap2 log_dens', 'snap2 pad1', 'snap2 log_pad1', 'snap2 pad2',
    'snap2 vx',      'snap2 vy',       'snap2 vz',
]
assert len(CHANNEL_NAMES) == 18

CONFIG_NAMES = ['Identity', 'Reflect-X', 'Reflect-Y', 'Reflect-Z']

# Velocity channels negated per reflection axis
VEL_CHANNEL_INDICES = [
    [7, 15],   # vx channels, negated when reflecting x
    [8, 16],   # vy channels, negated when reflecting y
    [9, 17],   # vz channels, negated when reflecting z
]

# ------------------------------------------------------------------
# Load one sub-box: raw shape is (N, 8, 8, 8, 18)
# After moveaxis -> (18, 8, 8, 8)
# ------------------------------------------------------------------
print(f"Loading sub-box {SUBBOX_IDX} from {DATA_PATH} ...")
dmo = np.load(DATA_PATH, mmap_mode='r')  # memory-map to avoid loading all
box = dmo[SUBBOX_IDX].astype(np.float32)         # (8, 8, 8, 18)
box = np.moveaxis(box, -1, 0)                     # (18, 8, 8, 8)
print(f"  Sub-box shape: {box.shape}")

# ------------------------------------------------------------------
# Build 4 configurations
# ------------------------------------------------------------------
def build_reflected(box, axis):
    """axis: 0=x, 1=y, 2=z  ->  flips spatial dim axis (dims 1,2,3 of box)."""
    d = np.flip(box, axis=axis + 1).copy()
    for ch in VEL_CHANNEL_INDICES[axis]:
        d[ch] *= -1
    return d

configs = [
    box.copy(),              # 0: identity
    build_reflected(box, 0), # 1: reflect-x
    build_reflected(box, 1), # 2: reflect-y
    build_reflected(box, 2), # 3: reflect-z
]

# Take z-slices: each config -> (18, 8, 8)
slices = [c[:, :, :, SLICE_IDX] for c in configs]   # list of (18, 8, 8)

# ------------------------------------------------------------------
# Compute global vmin/vmax per channel across all 4 configs
# (so the same panel uses the same color scale in every figure)
# ------------------------------------------------------------------
vmin = np.zeros(18)
vmax = np.zeros(18)
for ch in range(18):
    all_vals = np.concatenate([s[ch].ravel() for s in slices])
    vmin[ch] = np.nanmin(all_vals)
    vmax[ch] = np.nanmax(all_vals)
    # Avoid degenerate color range
    if vmin[ch] == vmax[ch]:
        vmin[ch] -= 1e-6
        vmax[ch] += 1e-6

# For signed channels (velocity), use symmetric range
for ax_idx, chs in enumerate(VEL_CHANNEL_INDICES):
    for ch in chs:
        absmax = max(abs(vmin[ch]), abs(vmax[ch]))
        vmin[ch], vmax[ch] = -absmax, absmax

# ------------------------------------------------------------------
# Plot: 4 figures, each with 18 panels (3 rows × 6 cols)
# ------------------------------------------------------------------
NROWS, NCOLS = 3, 6
FIGSIZE = (18, 9)

for cfg_idx, (sl, cfg_name) in enumerate(zip(slices, CONFIG_NAMES)):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE)
    fig.suptitle(
        f'Sub-box {SUBBOX_IDX} | Config: {cfg_name} | z-slice {SLICE_IDX}',
        fontsize=14, y=1.01,
    )

    for ch in range(18):
        row, col = ch // NCOLS, ch % NCOLS
        ax = axes[row, col]

        # Choose colormap: symmetric signed → RdBu_r; non-negative → viridis
        if vmin[ch] < 0 and vmax[ch] > 0:
            cmap = 'RdBu_r'
        else:
            cmap = 'viridis'

        im = ax.imshow(
            sl[ch],
            origin='lower',
            cmap=cmap,
            vmin=vmin[ch],
            vmax=vmax[ch],
            interpolation='nearest',
        )
        ax.set_title(f'ch{ch}: {CHANNEL_NAMES[ch]}', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'reflect_config_{cfg_idx}_{cfg_name.lower().replace("-", "_")}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

print("\nDone. Figures saved to:", OUT_DIR)