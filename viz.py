from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from config import A1_RESULTS_PATH, VIZ_DIR
from util import ece

VIZ_DIR = Path(VIZ_DIR)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times'], 'font.size': 13})


# HELPERS --------------------------------------------------------------------------------------------------------------


def _reliability_panel(ax, probs, labels, title, n_bins=10):
    """Draw one reliability diagram panel on ax."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bw = 1.0 / n_bins
    bin_centers = bins[:-1] + bw / 2.0

    accs = np.full(n_bins, np.nan)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.sum():
            accs[i] = labels[mask].mean()

    ece_val = ece(probs, labels, n_bins)
    valid = ~np.isnan(accs)

    # Accuracy bars
    ax.bar(bin_centers[valid], accs[valid], width=bw * 0.9,
           color='#4c72b0', alpha=0.85, label='Model outputs', zorder=2)

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration', zorder=3)

    # I-beam gap indicators: span from bar top (accs[i]) to diagonal (bin_centers[i])
    cap_w = bw * 0.25
    gap_color = '#ff814f'
    for i in np.where(valid)[0]:
        y_bar  = accs[i]
        y_diag = bin_centers[i]
        if abs(y_bar - y_diag) < 1e-6:
            continue
        x = bin_centers[i]
        ax.plot([x, x],                   [y_bar, y_diag], color=gap_color, lw=1.8, zorder=4)
        ax.plot([x - cap_w, x + cap_w],   [y_bar, y_bar],  color=gap_color, lw=1.8, zorder=4)
        ax.plot([x - cap_w, x + cap_w],   [y_diag, y_diag], color=gap_color, lw=1.8, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = np.arange(0, 1.01, 0.1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, color='#cccccc', linewidth=0.6, zorder=0)
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Fraction of positives', fontsize=14)
    ax.set_title(title, fontsize=15, pad=6)
    ax.set_aspect('equal', adjustable='box')

    legend = ax.legend(handles=[
        mlines.Line2D([], [], color='#4c72b0', lw=8, alpha=0.85, label='Model outputs'),
        mlines.Line2D([], [], color=gap_color, lw=1.8, label='Calibration gap'),
        mlines.Line2D([], [], color='black', lw=1.2, linestyle='--', label='Perfect calibration'),
    ], loc='upper left', fontsize=12, handlelength=2.4, borderpad=0.8)

    # Place ECE box just below the legend
    legend.figure.canvas.draw()
    leg_bb = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(leg_bb.x0, leg_bb.y0 - 0.02, f'ECE = {ece_val:.4f}',
            transform=ax.transAxes, fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#aaaaaa', alpha=0.9))


# PLOTS ----------------------------------------------------------------------------------------------------------------


def plot_reliability_diagram():
    data = dict(np.load(A1_RESULTS_PATH))
    logits = data['logits']
    T = float(data['temperature'])
    labels = data['labels']

    probs_before = 1.0 / (1.0 + np.exp(-logits))
    probs_after  = 1.0 / (1.0 + np.exp(-logits / T))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _reliability_panel(axes[0], probs_before, labels, 'Before temperature scaling')
    _reliability_panel(axes[1], probs_after,  labels, 'After temperature scaling')

    fig.suptitle('Approach 1: Reliability Diagram (test set)', fontsize=18)
    plt.tight_layout()

    out = VIZ_DIR / 'reliability_diagram.pdf'
    VIZ_DIR.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved:  {out}')


def main():
    plot_reliability_diagram()


if __name__ == '__main__':
    main()
