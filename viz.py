from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from config import A1_RESULTS_PATH, A2_RESULTS_PATH, VIZ_DIR, COST_RATIOS
from util import ece, EC, elkan_optimal_threshold, sigma_star


VIZ_DIR = Path(VIZ_DIR)
TITLE_FONT_SIZE = 16

# Times New Roman for all text; Computer Modern (CM) for math via mathtext
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times'],
    'font.size': 13,
    'mathtext.fontset': 'cm',  # classic LaTeX font
})


def plot_reliability_diagram():
    data = dict(np.load(A1_RESULTS_PATH))
    logits = data['logits']
    labels = data['labels']
    T = float(data['temperature'])

    # Apply sigmoid to logits to get probabilities (model confidence)
    probs_before = 1 / (1 + np.exp(-logits))
    probs_after  = 1 / (1 + np.exp(-logits / T))

    # Create two side-by-side diagrams to show reliability before and after calibration
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    n_bins = 10
    for i in range(2):
        # Two passes: before and after calibration
        ax = axes[i]
        probs = [probs_before, probs_after][i]
        title = ['Before', 'After'][i] + ' temperature scaling'

        bins = np.linspace(0, 1, n_bins + 1)
        bin_width = 1 / n_bins
        bin_centers = bins[:-1] + bin_width / 2

        # Compute fraction of positives per confidence bin (last bin includes upper boundary)
        accs = np.full(n_bins, np.nan)
        for j in range(n_bins):
            if j < n_bins - 1:
                mask = (probs >= bins[j]) & (probs < bins[j + 1])
            else:
                mask = (probs >= bins[j]) & (probs <= bins[j + 1])
            if mask.sum() != 0:
                accs[j] = labels[mask].mean()

        ece_val = ece(probs, labels, n_bins)
        valid = ~np.isnan(accs)

        # Accuracy bars
        ax.bar(bin_centers[valid], accs[valid], width=0.9*bin_width, color='#4c72b0', alpha=0.85, label='Model outputs', zorder=2)

        # Perfect calibration reference
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration', zorder=3)

        # I-beam gap indicators span from bar top (accs[i]) to diagonal (bin_centers[i])
        cap_width = bin_width * 0.25
        gap_color = '#ff814f'
        for bin_idx in np.where(valid)[0]:
            y_bar = accs[bin_idx]
            y_diag = bin_centers[bin_idx]
            if abs(y_bar - y_diag) < 1e-6:
                # Essentially no gap; model is perfectly calibrated here
                continue
            x = bin_centers[bin_idx]
            ax.plot([x, x], [y_bar, y_diag], color=gap_color, lw=1.8, zorder=4)
            ax.plot([x - cap_width, x + cap_width], [y_bar, y_bar], color=gap_color, lw=1.8, zorder=4)
            ax.plot([x - cap_width, x + cap_width], [y_diag, y_diag], color=gap_color, lw=1.8, zorder=4)

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
        ax.text(
            leg_bb.x0 + 0.011, leg_bb.y0 - 0.02, f'ECE = {ece_val:.5f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#aaaaaa', alpha=0.9)
        )

    fig.suptitle('Approach 1: Reliability Diagram (test set)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    plt.tight_layout()

    out = VIZ_DIR / 'reliability_diagram.pdf'
    VIZ_DIR.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved:  {out}')


def plot_ec_curves():
    a1 = dict(np.load(A1_RESULTS_PATH))
    a2 = dict(np.load(A2_RESULTS_PATH))
    labels = a1['labels']

    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    # EC values for A1, A2, and both trivial baselines across λ
    lams = COST_RATIOS
    lam_dense = np.logspace(np.log10(lams[0]), np.log10(lams[-1]), 1000)
    ec_a1 = [EC(a1[f'preds_{lam}'], labels, lam) for lam in lams]
    ec_a2 = [EC(a2[f'preds_{lam}'], labels, lam) for lam in lams]
    ec_pos = n_neg / n              # always shock (constant)
    ec_neg = lam_dense * n_pos / n  # never shock (linear)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Trivial baselines, plotted as dense continuous curves since they are closed-form functions
    ax.axhline(ec_pos, color='#888888', lw=1.5, linestyle='--', label=r'Always shock;  $\text{EC}\!^+\!\!(\lambda) = \text{P}(Y=0)$', zorder=2)
    ax.plot(lam_dense, ec_neg, ':', color='#888888', lw=1.5, label=r'Never shock;  $\text{EC}\!^-\!\!(\lambda) = \lambda \cdot \text{P}(Y=1)$', zorder=2)

    # A1 and A2 (primary focus of the plot)
    ax.plot(lams, ec_a1, 'o-', color='red', lw=2, ms=6, label=r'A1 (cost-sensitive thresholding)', zorder=3)
    ax.plot(lams, ec_a2, 's-', color='blue', lw=2, ms=6, label=r'A2 (cost-sensitive training)', zorder=3)

    # Axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(lams)
    ax.set_xticklabels([str(int(l)) if l == int(l) else str(l) for l in lams])
    ax.set_xlim(0.85, 1.2*lams[-1])
    ax.set_ylim(0.1, 1.4)
    yticks = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks])
    ax.set_xlabel(r'Cost ratio $\lambda$', fontsize=14)
    ax.set_ylabel(r'EC($\lambda$)', fontsize=14, rotation=0, labelpad=35)
    ax.set_title('Expected Cost vs. Cost Ratio (test set) (log-log scale)', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, color='#cccccc', linewidth=0.6, which='major', zorder=0)

    plt.tight_layout()
    out = VIZ_DIR / 'ec_curves.pdf'
    VIZ_DIR.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved:  {out}')


def plot_elkan_curve():
    # Get dense λ values and corresponding θ*(λ) values
    highlight_lams = [1, 2, 20]
    dense_lam_range = np.linspace(0, 25, 1000)
    theta_curve = elkan_optimal_threshold(dense_lam_range)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # Plot one panel per highlight λ; draw the θ*(λ) curve and annotate the specific threshold
    for ax, lam in zip(axes, highlight_lams):
        theta = elkan_optimal_threshold(lam)

        # θ*(λ) curve
        ax.plot(dense_lam_range, theta_curve, color='black', lw=1.8, zorder=3)

        # Red translucent shading above θ*(λ) (the "predict positive" region, i.e. SHOCKABLE)
        ax.axhspan(theta, 1.0, color='red', alpha=0.12, zorder=1)

        # Red horizontal threshold line
        ax.axhline(theta, color='red', lw=1.6, linestyle='--', zorder=4)

        # Red dot at (λ, θ*(λ))
        ax.plot(lam, theta, 'o', color='red', ms=4, zorder=5)

        # Vertical drop line from (λ, 0) to (λ, θ*(λ))
        ax.plot([lam, lam], [0, theta], color='#555555', lw=1.2, linestyle='--', zorder=2)

        # y-axis ticks: 0, threshold value (red + bold), 1
        # For λ=20, the θ=0.048 label is too close to 0 to fit as a tick label; use a free text annotation shifted up instead
        if lam == 20:
            ax.set_yticks([0, theta, 1])
            ax.set_yticklabels(['0', '', '1'], fontsize=12)
            ax.text(-0.01, theta - 0.02, f'{theta:.3f}', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=12, color='red', fontweight='bold')
        else:
            ax.set_yticks([0, theta, 1])
            if lam == 1:
                theta_label = f'{theta:.1f}'
            else:
                theta_label = f'{theta:.3f}'
            tick_labels = ax.set_yticklabels(['0', theta_label, '1'], fontsize=12)
            tick_labels[1].set_color('red')
            tick_labels[1].set_fontweight('bold')
        ax.set_ylim(0, 1)

        # λ annotation: below x-axis for λ=1,2; just above curve point for λ=20
        if lam == 20:
            ax.text(lam, theta + 0.06, rf'$\lambda={lam}$', ha='center', va='bottom', fontsize=12, color='#555555')
        else:
            ax.text(lam, -0.07, rf'$\lambda={lam}$', ha='center', va='top', fontsize=12, color='#555555', transform=ax.get_xaxis_transform())

        # Region labels on the first panel only (SHOCKABLE / NON-SHOCKABLE)
        if lam == highlight_lams[0]:
            label_color = '#aaaaaa'
            ax.text(12.5, (theta + 1) / 2, 'SHOCKABLE', ha='center', va='center', fontsize=15, color=label_color, fontweight='bold', zorder=2)
            ax.text(12.5, theta / 2, 'NON-SHOCKABLE', ha='center', va='center', fontsize=15, color=label_color, fontweight='bold', zorder=2)

        ax.set_xlim(0, 25)
        ax.grid(True, color='#cccccc', linewidth=0.6, zorder=0)
        ax.set_ylabel(r'$\theta\!^*\!(\lambda)$', fontsize=13, rotation=0)

    x_ticks = range(0, 25+1, 5)  # 0, 5, ..., 25
    axes[-1].set_xlabel(r'Cost ratio $\lambda$', fontsize=14)
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xticklabels([str(x) for x in x_ticks], fontsize=12)
    fig.suptitle(r"Elkan's Optimal Threshold:  $\theta\!^*\!(\lambda) = \dfrac{1}{\lambda+1}$", fontsize=TITLE_FONT_SIZE, fontweight='bold')

    # Pin all y-axis labels to the same x so they align regardless of varying tick label widths (e.g. "0.5" vs "0.333")
    for ax in axes:
        ax.yaxis.set_label_coords(-0.07, 0.5)
    plt.tight_layout()

    out = VIZ_DIR / 'elkan_curve.pdf'
    VIZ_DIR.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved:  {out}')


def plot_sigma_star_curve():
    highlight_lams = [1, 2, 20]
    p_range = np.linspace(0, 1, 1000)

    # sharey=True only; x-ticks differ per panel due to the various "highlight_lams", so sharex=False
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=False, sharey=True)
    fig.subplots_adjust(wspace=0.08)

    # Plot one panel per highlight λ; draw the σ*(p) curve and annotate the decision boundary at p=θ*(λ)
    for i, (ax, lam) in enumerate(zip(axes, highlight_lams)):
        theta = elkan_optimal_threshold(lam)
        sigma_curve = sigma_star(p_range, lam)

        # σ*(p) curve
        ax.plot(p_range, sigma_curve, color='blue', lw=1.8, zorder=3)

        # Red translucent shading for p > θ*(λ) (the "predict positive" region, i.e. SHOCKABLE)
        ax.axvspan(theta, 1.0, color='red', alpha=0.12, zorder=1)

        # Red vertical dashed line at p=θ*(λ) (the effective boundary)
        ax.axvline(theta, color='red', lw=1.6, linestyle='--', zorder=4)

        # Red dot at (θ*(λ), 0.5)
        ax.plot(theta, 0.5, 'o', color='red', ms=4, zorder=5)

        # Horizontal gray dashed line at y=0.5 from x=0 to x=θ*(λ) (fixed A2 decision threshold = 0.5)
        ax.plot([0, theta], [0.5, 0.5], color='#555555', lw=1.2, linestyle='--', zorder=2)

        # x-ticks: 0, θ*(λ) in red+bold, 1
        # For λ=20, the θ=0.048 label is too close to 0; use a free text annotation shifted right instead
        if lam == 20:
            ax.set_xticks([0, theta, 1])
            ax.set_xticklabels(['0', '', '1'], fontsize=12)
            ax.text(theta, -0.037, f'{theta:.3f}', transform=ax.get_xaxis_transform(), ha='left', va='top', fontsize=12, color='red', fontweight='bold')
        else:
            ax.set_xticks([0, theta, 1])
            if lam == 1:
                theta_label = f'{theta:.1f}'
            else:
                theta_label = f'{theta:.3f}'
            x_tick_labels = ax.set_xticklabels(['0', theta_label, '1'], fontsize=12)
            x_tick_labels[1].set_color('red')
            x_tick_labels[1].set_fontweight('bold')

        # y-ticks only on leftmost panel (sharey suppresses the rest automatically)
        if i == 0:
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '0.5', '1'], fontsize=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, color='#cccccc', linewidth=0.6, zorder=0)
        ax.set_title(rf'$\lambda = {lam}$', fontsize=14, fontweight='bold')
        ax.set_xlabel('True probability $p$', fontsize=13)

        # Region labels on the first panel only, rotated 45 degrees CCW, centered at y=0.5
        if i == 0:
            label_color = '#aaaaaa'
            ax.text((theta + 1) / 2, 0.5, 'SHOCKABLE', ha='center', va='center', fontsize=12, color=label_color, fontweight='bold', rotation=45, zorder=2)
            ax.text(theta / 2, 0.5, 'NON-SHOCKABLE', ha='center', va='center', fontsize=12, color=label_color, fontweight='bold', rotation=45, zorder=2)

    axes[0].set_ylabel(r'$\sigma\!^*\!(p)$', fontsize=14, rotation=0, labelpad=28)
    fig.suptitle(
        r"Optimal Confidence Under Cost-Sensitive Training:  $\sigma\!^*\!(p) = \dfrac{\lambda p}{1 - p + \lambda p}$",
        fontsize=TITLE_FONT_SIZE,
        fontweight='bold'
    )

    plt.tight_layout()
    out = VIZ_DIR / 'sigma_star_curve.pdf'
    VIZ_DIR.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved:  {out}')


def main():
    plot_reliability_diagram()
    plot_ec_curves()
    plot_elkan_curve()
    plot_sigma_star_curve()


if __name__ == '__main__':
    main()
