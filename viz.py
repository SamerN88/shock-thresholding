from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from config import A1_RESULTS_PATH, A2_RESULTS_PATH, VIZ_DIR, COST_RATIOS
from util import ece, EC, elkan_optimal_threshold


# TODO: review this file


VIZ_DIR = Path(VIZ_DIR)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times'],
    'font.size': 13,
    'mathtext.fontset': 'cm',  # Computer Modern (classic LaTeX math font)
})


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


def plot_ec_curve():
    a1 = dict(np.load(A1_RESULTS_PATH))
    a2 = dict(np.load(A2_RESULTS_PATH))
    labels = a1['labels']

    n     = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    lams   = COST_RATIOS
    ec_a1  = [EC(a1[f'preds_{lam}'], labels, lam) for lam in lams]
    ec_a2  = [EC(a2[f'preds_{lam}'], labels, lam) for lam in lams]
    ec_pos = n_neg / n                              # constant: always shock
    ec_neg = [lam * n_pos / n for lam in lams]     # linear:   never shock
    lam_star = n_neg / n_pos                        # crossover: EC+ = EC−

    fig, ax = plt.subplots(figsize=(8, 5))

    # Trivial baselines, plotted as dense continuous curves since they are closed-form functions
    lam_dense = np.logspace(np.log10(lams[0]), np.log10(lams[-1]), 300)
    ax.axhline(ec_pos, color='#999999', lw=1.5, linestyle='--', label='Always shock (EC+)', zorder=2)
    ax.plot(lam_dense, lam_dense * n_pos / n, ':', color='#999999', lw=1.5, label='Never shock (EC−)', zorder=2)

    # A1 and A2 (primary)
    ax.plot(lams, ec_a1, 'o-', color='red', lw=2.0, ms=6, label=r'A1 (cost-sensitive thresholding)', zorder=3)
    ax.plot(lams, ec_a2, 's-', color='blue', lw=2.0, ms=6, label=r'A2 (cost-sensitive training)',    zorder=3)

    # Crossover annotation
    ax.axvline(lam_star, color='#cccccc', lw=1.0, linestyle='--', zorder=1)
    ax.text(lam_star * 1.06, 0.02, rf'$\lambda\!^* \approx$ {lam_star:.1f}', transform=ax.get_xaxis_transform(), fontsize=11, color='#888888', va='bottom')

    # Axes
    ax.set_xscale('log')
    ax.set_xticks(lams)
    ax.set_xticklabels([str(int(l)) if l == int(l) else str(l) for l in lams])
    ax.set_xlim(0.85, 24)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel(r'Cost ratio $\lambda$', fontsize=14)
    ax.set_ylabel(r'EC($\lambda$)', fontsize=14, rotation=0, labelpad=35)
    ax.set_title('Expected Cost vs. Cost Ratio (test set)', fontsize=15)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, color='#cccccc', linewidth=0.6, which='major', zorder=0)

    plt.tight_layout()
    out = VIZ_DIR / 'ec_curve.pdf'
    VIZ_DIR.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved:  {out}')


def plot_elkan_curve():
    """
    Three stacked panels showing Elkan's optimal threshold θ*(λ) = 1/(1+λ).
    Each panel highlights a specific λ from COST_RATIOS with a vertical drop line,
    a horizontal threshold line, and a red-shaded "predict positive" region above θ*(λ).
    """

    highlight_lams = [1, 2, 20]
    lam_range = np.linspace(0, 25, 1000)
    theta_curve = elkan_optimal_threshold(lam_range)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for ax, lam_val in zip(axes, highlight_lams):
        theta_val = elkan_optimal_threshold(lam_val)

        # θ*(λ) curve
        ax.plot(lam_range, theta_curve, color='black', lw=1.8, zorder=3)

        # Red translucent shading above θ*(λ_val) — "predict positive" region
        ax.axhspan(theta_val, 1.0, color='red', alpha=0.12, zorder=1)

        # Horizontal threshold line — red dashed
        ax.axhline(theta_val, color='red', lw=1.6, linestyle='--', zorder=2)

        # Vertical drop line from (λ_val, 0) to (λ_val, θ*(λ_val))
        ax.plot([lam_val, lam_val], [0, theta_val], color='#555555', lw=1.2, linestyle='--', zorder=2)

        # Dot at (λ_val, θ*(λ_val))
        ax.plot(lam_val, theta_val, 'o', color='black', ms=5, zorder=4)

        # Y-axis ticks: 0, threshold value (red + bold), 1
        # For λ=20 θ*≈0.048 is too close to 0 to fit as a tick label — use a text annotation instead
        if lam_val == 20:
            ax.set_yticks([0, theta_val, 1])
            ax.set_yticklabels([f'{0:.1f}', '', f'{1:.1f}'], fontsize=12)
            ax.text(-0.01, theta_val - 0.02, f'{theta_val:.3f}',
                    transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                    fontsize=12, color='red', fontweight='bold')
        else:
            ax.set_yticks([0, theta_val, 1])
            if lam_val == 1:
                theta_label = '0.5'
            else:
                theta_label = f'{theta_val:.3f}'
            tick_labels = ax.set_yticklabels([f'{0:.1f}', theta_label, f'{1:.1f}'], fontsize=12)
            tick_labels[1].set_color('red')
            tick_labels[1].set_fontweight('bold')
        ax.set_ylim(0, 1)

        # λ annotation: below x-axis for λ=1,2; just above curve point for λ=20 (no space below)
        if lam_val == 20:
            ax.text(lam_val, theta_val + 0.06, rf'$\lambda$={lam_val}', ha='center', va='bottom', fontsize=12, color='#555555')
        else:
            ax.text(lam_val, -0.07, rf'$\lambda$={lam_val}', ha='center', va='top', fontsize=12, color='#555555', transform=ax.get_xaxis_transform())

        # Region labels on the first panel only
        if lam_val == highlight_lams[0]:
            label_color = '#aaaaaa'
            ax.text(12.5, (theta_val + 1.0) / 2, 'SHOCKABLE',
                    ha='center', va='center', fontsize=15, color=label_color,
                    fontweight='bold', zorder=2)
            ax.text(12.5, theta_val / 2, 'NON-SHOCKABLE',
                    ha='center', va='center', fontsize=15, color=label_color,
                    fontweight='bold', zorder=2)

        ax.set_xlim(0, 25)
        ax.grid(True, color='#cccccc', linewidth=0.6, zorder=0)
        ax.set_ylabel(r'$\theta\!^*\!(\lambda)$', fontsize=13, rotation=0)

    axes[-1].set_xlabel(r'Cost ratio $\lambda$', fontsize=14)
    axes[-1].set_xticks(range(0, 26, 5))
    fig.suptitle(r"Elkan's Optimal Threshold $\theta\!^*\!(\lambda) = \dfrac{1}{\lambda+1}$", fontsize=16)

    # Pin all y-axis labels to the same x so they align regardless of
    # varying tick label widths ("0.5" vs "0.333" vs "0.048").
    # Value is in axes coordinates; negative = left of the axes edge.
    for ax in axes:
        ax.yaxis.set_label_coords(-0.07, 0.5)
    plt.tight_layout()

    out = VIZ_DIR / 'elkan_curve.pdf'
    VIZ_DIR.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved:  {out}')


def main():
    plot_reliability_diagram()
    plot_ec_curve()
    plot_elkan_curve()


if __name__ == '__main__':
    main()
