import numpy as np
import matplotlib.pyplot as plt

# MNRAS journal style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
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

data = np.load('/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits.npz')
wte = data['wte']   # (vocab_size, n_embd)

wte_norm = wte / np.linalg.norm(wte, axis=1, keepdims=True)
similarity = wte_norm @ wte_norm[20]   # (vocab_size,)

token_idx = np.arange(len(similarity))

# Non-uniform x mapping: stretch indices 132-135 by `stretch` factor
SPLIT   = 132   # last regular token index
STRETCH = 8     # how many display units each special token occupies

def rx(x):
    """Remap raw token index to stretched display coordinate."""
    return np.where(np.asarray(x) <= SPLIT,
                    x,
                    SPLIT + (np.asarray(x) - SPLIT) * STRETCH)

x_plot = rx(token_idx)

# MNRAS single-column width ~3.4 in
fig, ax = plt.subplots(figsize=(3.4, 2.4))

ax.plot(x_plot[:132], similarity[:132], color='tab:blue', linewidth=1,
        rasterized=True, markersize=2, marker='o',label='VALUE')
ax.axvspan(rx(0), rx(39), color='tab:blue', alpha=0.12, linewidth=0)
ax.text(rx(20), 0.97, 'position token',
        ha='center', va='top', fontsize=6, color='tab:blue',
        transform=ax.get_xaxis_transform())

# Dotted vertical line to mark the axis break
#ax.axvline(SPLIT, color='gray', linewidth=0.6, linestyle=':')

special_tokens = [
    (132, 'START', '#e41a1c', 'o'),
    (133, 'SPACE', '#ff7f00', 's'),
    (134, 'PAD',   '#4daf4a', '^'),
    (135, 'END',   '#984ea3', 'D'),
]
for idx, name, color, marker in special_tokens:
    ax.scatter(rx(idx), similarity[idx], color=color, marker=marker,
               s=10, zorder=3, label=name)

ax.axhline(0, color='black', linestyle='--', linewidth=0.6)
ax.legend(loc='upper right', framealpha=0., edgecolor='0.8')
ax.set_xlabel('Token index')
ax.set_ylabel('Cosine similarity to $\\mathbf{e}_{20}$')

# Restore original token-index labels on the x-axis
tick_raw = [0, 20, 40, 60, 80, 100, 120, 132, 133, 134, 135]
ax.set_xticks(rx(np.array(tick_raw)))
ax.set_xticklabels(tick_raw)
x_range = rx(token_idx[-1]) - rx(token_idx[0])
ax.set_xlim(rx(token_idx[0]) - 0.03 * x_range,
            rx(token_idx[-1]) + 0.03 * x_range)

plt.tight_layout()
plt.savefig('wte_similarity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('wte_similarity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved wte_similarity.pdf / .png")
