from types import SimpleNamespace

import numpy as np

from config import RANDOM_SEED, COST_RATIOS, A1_RESULTS_PATH, A2_RESULTS_PATH
from util import metrics, EC, Glyphs


# Special characters, just for pretty display
sH = Glyphs.sH
dH = Glyphs.dH


def paired_bootstrap_test(preds_left, preds_right, labels, cost_ratio, B=10000):
    """
    Paired bootstrap test on ΔEC(λ) = EC_left(λ) - EC_right(λ)

    Both models are evaluated on the same resampled indices each iteration, preserving
    the paired structure. The bootstrap distribution is shifted by the observed ΔEC(λ)
    (the one from the full dataset) to center it at 0 (null hypothesis: ΔEC = 0) for
    the two-sided p-value.

    Returns: (delta_obs, p_value, ci_low, ci_high)
        delta_obs       observed ΔEC(λ); negative means left is better
        p_value         two-sided p-value under H0
        ci_low/ci_high  2.5/97.5 percentiles of the bootstrap distribution
    """

    # Use fixed random seed for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(labels)

    # Get the observed EC(λ) on the whole dataset for both left and right
    ec_left_obs  = EC(preds_left,  labels, cost_ratio)
    ec_right_obs = EC(preds_right, labels, cost_ratio)
    delta_obs = ec_left_obs - ec_right_obs  # ΔEC(λ)

    # Draw B bootstrap samples, each of size n=len(dataset), and collect ΔEC(λ) for each bootstrap
    delta_boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ec_left_boot = EC(preds_left[idx], labels[idx], cost_ratio)
        ec_right_boot = EC(preds_right[idx], labels[idx], cost_ratio)
        delta_boot[b] = ec_left_boot - ec_right_boot

    # Shift bootstrap distribution to the null hypothesis center (ΔEC = 0), then compute two-sided p-value
    abs_diffs = np.abs(delta_boot - delta_obs)
    p_value = float(np.mean(abs_diffs >= abs(delta_obs)))
    ci_low, ci_high = np.percentile(delta_boot, [2.5, 97.5])  # 95% confidence interval

    return delta_obs, p_value, ci_low, ci_high


def compare_ec(cost_ratio, a1_data, a2_data):
    """
    Unified comparison for a single λ: prints one Performance Metrics table and one
    Statistical Comparisons table covering all five comparison pairs:
        - A1 vs. A2
        - A1 vs. Trivial Pos
        - A1 vs. Trivial Neg
        - A2 vs. Trivial Pos
        - A2 vs. Trivial Neg
    """

    print(f'λ = {cost_ratio}')
    print()

    labels = a1_data['labels']
    preds_a1 = a1_data[f'preds_{cost_ratio}']
    preds_a2 = a2_data[f'preds_{cost_ratio}']

    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    # Trivial baseline predictions (deterministic)
    preds_tp = np.ones(n, dtype=preds_a1.dtype)   # always predict positive; EC+ = P(Y=0)
    preds_tn = np.zeros(n, dtype=preds_a1.dtype)  # always predict negative; EC- = λ·P(Y=1)

    # Metrics
    ec_tp, fpr_tp, fnr_tp = n_neg / n, 1.0, 0.0
    ec_tn, fpr_tn, fnr_tn = cost_ratio * n_pos / n, 0.0, 1.0
    _, _, _, _, _, fpr1, fnr1 = metrics(preds_a1, labels)
    _, _, _, _, _, fpr2, fnr2 = metrics(preds_a2, labels)
    ec1 = EC(preds_a1, labels, cost_ratio)
    ec2 = EC(preds_a2, labels, cost_ratio)

    # Table 1: Performance Metrics -------------------------------------------------------------------------------------
    T1 = SimpleNamespace(w1=8,  w2=15, w3=15, w4=13, w5=13)

    sep1 = sH * (T1.w1 + T1.w2 + T1.w3 + T1.w4 + T1.w5)
    print('Performance Metrics'.center(len(sep1)))
    print(sep1)
    print(f'{"Metric":<{T1.w1}}{"Trivial Pos":>{T1.w2}}{"Trivial Neg":>{T1.w3}}{"A1":>{T1.w4}}{"A2":>{T1.w5}}')
    print(sep1)
    print(f'{"EC(λ)":<{T1.w1}}{ec_tp:>{T1.w2}.5f}{ec_tn:>{T1.w3}.5f}{ec1:>{T1.w4}.5f}{ec2:>{T1.w5}.5f}')
    print(f'{"FPR":<{T1.w1}}{fpr_tp:>{T1.w2}.5f}{fpr_tn:>{T1.w3}.5f}{fpr1:>{T1.w4}.5f}{fpr2:>{T1.w5}.5f}')
    print(f'{"FNR":<{T1.w1}}{fnr_tp:>{T1.w2}.5f}{fnr_tn:>{T1.w3}.5f}{fnr1:>{T1.w4}.5f}{fnr2:>{T1.w5}.5f}')
    print(sep1)
    # ------------------------------------------------------------------------------------------------------------------

    # Bootstrap tests for all five pairs
    d12, p12, lo12, hi12 = paired_bootstrap_test(preds_a1, preds_a2, labels, cost_ratio)
    d1p, p1p, lo1p, hi1p = paired_bootstrap_test(preds_a1, preds_tp, labels, cost_ratio)
    d1n, p1n, lo1n, hi1n = paired_bootstrap_test(preds_a1, preds_tn, labels, cost_ratio)
    d2p, p2p, lo2p, hi2p = paired_bootstrap_test(preds_a2, preds_tp, labels, cost_ratio)
    d2n, p2n, lo2n, hi2n = paired_bootstrap_test(preds_a2, preds_tn, labels, cost_ratio)

    # Table 2: Statistical Comparisons ---------------------------------------------------------------------------------
    T2 = SimpleNamespace(w1=22, w2=11, w3=24, w4=9)

    def fmt_p(p):
        return '< 0.001' if p < 0.001 else f'{p:.3f}'

    def fmt_row(label, delta, ci_lo, ci_hi, p):
        ci_str = f'[{ci_lo:+.5f}, {ci_hi:+.5f}]'
        return f'{label:<{T2.w1}}{delta:>+{T2.w2}.5f}    {ci_str:<{T2.w3}}{fmt_p(p):>{T2.w4}}'

    sep2 = sH * (T2.w1 + T2.w2 + 4 + T2.w3 + T2.w4)
    verdict = 'A1 wins' if ec1 < ec2 else ('A2 wins' if ec2 < ec1 else 'tie')
    print()
    print('Statistical Comparisons'.center(len(sep2)))
    print(sep2)
    print(f'{"Comparison":<{T2.w1}}{"ΔEC":>{T2.w2}}    {"95% CI":<{T2.w3}}{"p   ":>{T2.w4}}')
    print(sep2)
    print(fmt_row('A1 vs. A2', d12, lo12, hi12, p12) + f'  ({verdict})')
    print(fmt_row('A1 vs. Trivial Pos', d1p, lo1p, hi1p, p1p))
    print(fmt_row('A1 vs. Trivial Neg', d1n, lo1n, hi1n, p1n))
    print(fmt_row('A2 vs. Trivial Pos', d2p, lo2p, hi2p, p2p))
    print(fmt_row('A2 vs. Trivial Neg', d2n, lo2n, hi2n, p2n))
    print(sep2)
    print(f'Note:  ΔEC = left - right')
    # ------------------------------------------------------------------------------------------------------------------


def main():
    a1_data = dict(np.load(A1_RESULTS_PATH))
    a2_data = dict(np.load(A2_RESULTS_PATH))

    for cost_ratio in COST_RATIOS:
        print(dH*100)
        compare_ec(cost_ratio, a1_data, a2_data)
        print()
    print(dH*100)


if __name__ == '__main__':
    main()
