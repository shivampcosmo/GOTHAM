import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

# MNRAS journal style (matches plot_logits.py)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 6,
    'axes.titlesize': 7,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 4,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

NHALO = 3
NPROP = 8
LABELS = [r'$x$', r'$y$', r'$z$', 'mass', r'$v_x$', r'$v_y$', r'$v_z$', 'concentration']
COLORS = ['tab:blue', 'tab:orange', 'tab:green']


def make_logits_figure_single(idx, logits, sentences, truth_sentences, outdir='.'):
    """Save a logits figure for a single sample index (one sub-box per panel)."""
    logits_sel = logits[idx]  # (L-1, vocab_size)
    truth_sel = truth_sentences[idx]  # (L,)
    prop = np.exp(logits_sel) / np.exp(logits_sel).sum(axis=-1, keepdims=True)  # (L-1, vocab_size)
    prop = prop[6:6 + NHALO * NPROP].reshape((NHALO, NPROP, -1))  # (nhalo, nprop, vocab_size)
    truth_sel = truth_sel[7:7 + NHALO * NPROP].reshape((NHALO, NPROP))  # (nhalo, nprop)

    fig = plt.figure(figsize=(7.0, 2.4))
    outer = gridspec.GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.5)

    for i in range(NPROP):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(outer[row, col])
        for h in range(NHALO):
            label = f"Halo {h + 1}" if i == 3 else None
            ax.plot(prop[h, i, :], color=COLORS[h], linewidth=0.5, label=label)
        ymin, ymax = ax.get_ylim()
        for h in range(NHALO):
            ax.vlines(x=truth_sel[h, i], ymin=ymin, ymax=prop[:, i, :].max(),
                      color=COLORS[h], linewidth=0.5)
        ax.set_ylim(ymin, ymax)
        if i in (0, 1, 2):
            ax.set_xlim(0, 41)
        ax.set_title(LABELS[i])
        if row == 1:
            ax.set_xlabel("Token index", labelpad=1)
        if col == 0:
            ax.set_ylabel("Probability")
        if i == 3:
            ax.legend(loc='upper right', framealpha=0., edgecolor='0.8')

    fig.tight_layout()
    fig.savefig(f"{outdir}/logits_halo_{idx}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(f"{outdir}/logits_halo_{idx}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved logits_halo_{idx}.pdf / .png")


if __name__ == "__main__":
    data = np.load('/work/hdd/bdne/yzhang116/cp_visualize/during_gen/sim66_embeddings_logits_blocks.npz')
    truth = np.load('/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits_blocks.npz')
    logits = data['logits_100']
    sentences = data['sentence']
    truth_sentences = truth['sentence']

    idx = int(sys.argv[1])
    make_logits_figure_single(idx, logits, sentences, truth_sentences)
