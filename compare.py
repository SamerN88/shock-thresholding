import numpy as np

from config import RANDOM_SEED, COST_RATIOS, A1_RESULTS_PATH, A2_RESULTS_PATH
from util import metrics, MC


# TODO: review this file


def paired_bootstrap_test(preds_a1, preds_a2, labels, cost_ratio, B=10000):
    """
    Paired bootstrap test on ΔMC(λ) = MC_A1(λ) − MC_A2(λ)

    Both models are evaluated on the same resampled indices each iteration,
    preserving the paired structure. The bootstrap distribution is shifted to
    be centered at 0 (H0: ΔMC = 0) for the two-sided p-value.

    Returns: (delta_obs, p_value, ci_low, ci_high)
        delta_obs       observed ΔMC(λ); negative means A1 is better
        p_value         two-sided p-value under H0: ΔMC = 0
        ci_low/ci_high  2.5/97.5 percentiles of the bootstrap distribution
    """

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(labels)

    mc1_obs = MC(preds_a1, labels, cost_ratio)
    mc2_obs = MC(preds_a2, labels, cost_ratio)
    delta_obs = mc1_obs - mc2_obs

    delta_boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        mc1_b = MC(preds_a1[idx], labels[idx], cost_ratio)
        mc2_b = MC(preds_a2[idx], labels[idx], cost_ratio)
        delta_boot[b] = mc1_b - mc2_b

    # Shift bootstrap distribution to H0 (delta=0), then two-sided p-value
    p_value = float(np.mean(np.abs(delta_boot - delta_obs) >= abs(delta_obs)))
    ci_low, ci_high = np.percentile(delta_boot, [2.5, 97.5])

    return delta_obs, p_value, ci_low, ci_high


def compare(cost_ratio, a1_data, a2_data):
    labels   = a1_data['labels']
    preds_a1 = a1_data[f'preds_{cost_ratio}']
    preds_a2 = a2_data[f'preds_{cost_ratio}']

    _, tp1, fp1, tn1, fn1, fpr1, fnr1 = metrics(preds_a1, labels)
    _, tp2, fp2, tn2, fn2, fpr2, fnr2 = metrics(preds_a2, labels)
    mc1 = MC(preds_a1, labels, cost_ratio)
    mc2 = MC(preds_a2, labels, cost_ratio)
    delta_obs, p_value, ci_low, ci_high  = paired_bootstrap_test(preds_a1, preds_a2, labels, cost_ratio)

    winner = 'A1' if mc1 < mc2 else ('A2' if mc2 < mc1 else 'TIE')
    alpha  = 0.05

    print(f'Paired Bootstrap Test (B=10,000):  A1 vs A2  at  λ = {cost_ratio}')
    print(f'  {"Metric":<12} {"A1":>12} {"A2":>12}')
    print(f'  {"-"*38}')
    print(f'  {"MC(λ)":<12} {mc1:>12.5f} {mc2:>12.5f}    ({winner} wins)')
    print(f'  {"FPR":<12} {fpr1:>12.5f} {fpr2:>12.5f}')
    print(f'  {"FNR":<12} {fnr1:>12.5f} {fnr2:>12.5f}')
    print(f'  {"TP":<12} {tp1:>12} {tp2:>12}')
    print(f'  {"FP":<12} {fp1:>12} {fp2:>12}')
    print(f'  {"TN":<12} {tn1:>12} {tn2:>12}')
    print(f'  {"FN":<12} {fn1:>12} {fn2:>12}')
    print()
    print(f'  ΔMC(λ) = {delta_obs:+.5f}  (A1 − A2; negative = A1 better)')
    print(f'  95% CI: [{ci_low:+.5f},  {ci_high:+.5f}]')
    if p_value < 0.001:
        print(f'  p < 0.001')
    else:
        print(f'  p = {p_value:.4f}')
    print(f'  {"SIGNIFICANT" if p_value < alpha else "not significant"}  (α = {alpha})')


def main():
    a1_data = dict(np.load(A1_RESULTS_PATH))
    a2_data = dict(np.load(A2_RESULTS_PATH))

    for lam in COST_RATIOS:
        print('=' * 70)
        compare(lam, a1_data, a2_data)
        print()


if __name__ == '__main__':
    main()
