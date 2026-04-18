import numpy as np

from config import RANDOM_SEED, COST_RATIOS, A1_RESULTS_PATH, A2_RESULTS_PATH
from util import metrics, EC, Glyphs


# Special characters, just for pretty display
sH = Glyphs.sH
dH = Glyphs.dH


def compare_EC(cost_ratio, a1_data, a2_data):
    """
    Compares the EC(λ) of A1 vs. A2 to see if their difference is statistically significant,
    using paired bootstrap test.
    """

    labels   = a1_data['labels']
    preds_a1 = a1_data[f'preds_{cost_ratio}']
    preds_a2 = a2_data[f'preds_{cost_ratio}']

    # Get all necessary metrics for A1 and A2, and run paired bootstrap test for EC(λ)
    _, tp1, fp1, tn1, fn1, fpr1, fnr1 = metrics(preds_a1, labels)
    _, tp2, fp2, tn2, fn2, fpr2, fnr2 = metrics(preds_a2, labels)
    ec1 = EC(preds_a1, labels, cost_ratio)
    ec2 = EC(preds_a2, labels, cost_ratio)
    delta_obs, p_value, ci_low, ci_high  = paired_bootstrap_test(preds_a1, preds_a2, labels, cost_ratio, B=10000)

    winner = 'A1' if ec1 < ec2 else ('A2' if ec2 < ec1 else None)

    print(f'A1 vs. A2: Paired Bootstrap Test (B=10,000)')
    print(f'λ = {cost_ratio}')
    print(sH*38)
    print(f'{"Metric":<12} {"A1":>12} {"A2":>12}')
    print(sH*38)
    print(f'{"EC(λ)":<12} {ec1:>12.5f} {ec2:>12.5f}    ({"tie" if winner is None else f"{winner} wins"})')
    print(f'{"FPR":<12} {fpr1:>12.5f} {fpr2:>12.5f}')
    print(f'{"FNR":<12} {fnr1:>12.5f} {fnr2:>12.5f}')
    print(f'{"TP":<12} {tp1:>12} {tp2:>12}')
    print(f'{"FP":<12} {fp1:>12} {fp2:>12}')
    print(f'{"TN":<12} {tn1:>12} {tn2:>12}')
    print(f'{"FN":<12} {fn1:>12} {fn2:>12}')
    print()
    print(f'ΔEC(λ) = {delta_obs:+.5f}    (ΔEC = A1 - A2)')
    print(f'95% CI: [{ci_low:+.5f}, {ci_high:+.5f}]')
    if p_value < 0.001:
        print(f'p < 0.001')
    else:
        print(f'p = {p_value:.5f}')
    print(sH*38)


def paired_bootstrap_test(preds_a1, preds_a2, labels, cost_ratio, B=10000):
    """
    Paired bootstrap test on ΔEC(λ) = EC_A1(λ) - EC_A2(λ)

    Both models are evaluated on the same resampled indices each iteration, preserving
    the paired structure. The bootstrap distribution is shifted by the observed ΔEC(λ)
    (the one from the full dataset) to center it at 0 (null hypothesis: ΔEC = 0) for
    the two-sided p-value.

    Returns: (delta_obs, p_value, ci_low, ci_high)
        delta_obs       observed ΔEC(λ); negative means A1 is better
        p_value         two-sided p-value under H0
        ci_low/ci_high  2.5/97.5 percentiles of the bootstrap distribution
    """

    # Use fixed random seed for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(labels)

    # Get the observed EC(λ) on the whole dataset for both A1 and A2, and the ΔEC(λ)
    ec1_obs = EC(preds_a1, labels, cost_ratio)
    ec2_obs = EC(preds_a2, labels, cost_ratio)
    delta_obs = ec1_obs - ec2_obs

    # Draw B bootstrap samples, each of size n=len(dataset), and collect ΔEC(λ) for each bootstrap
    delta_boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ec1_b = EC(preds_a1[idx], labels[idx], cost_ratio)
        ec2_b = EC(preds_a2[idx], labels[idx], cost_ratio)
        delta_boot[b] = ec1_b - ec2_b

    # Shift bootstrap distribution to the null hypothesis center (ΔEC = 0), then compute two-sided p-value
    abs_diffs = np.abs(delta_boot - delta_obs)
    p_value = float(np.mean(abs_diffs >= abs(delta_obs)))
    ci_low, ci_high = np.percentile(delta_boot, [2.5, 97.5])  # 95% confidence interval

    return delta_obs, p_value, ci_low, ci_high


def compare_trivial(cost_ratio, a1_data, a2_data):
    """
    Compares A1 and A2 against the two trivial baselines at a given λ:
        EC+(λ) = P(Y=0)     always predict positive (θ=0); FPR=1, FNR=0
        EC-(λ) = λ·P(Y=1)   always predict negative (θ=1); FPR=0, FNR=1
    """

    labels = a1_data['labels']
    preds_a1 = a1_data[f'preds_{cost_ratio}']
    preds_a2 = a2_data[f'preds_{cost_ratio}']

    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    # Trivial baseline ECs
    ec_pos  = n_neg / n                 # always predict positive: EC+ = P(Y=0)
    ec_neg = cost_ratio * n_pos / n     # always predict negative: EC- = λ·P(Y=1)

    _, _, _, _, _, fpr1, fnr1 = metrics(preds_a1, labels)
    _, _, _, _, _, fpr2, fnr2 = metrics(preds_a2, labels)
    ec1 = EC(preds_a1, labels, cost_ratio)
    ec2 = EC(preds_a2, labels, cost_ratio)

    print(f'Trivial Baselines')
    print(f'λ = {cost_ratio}')
    print(sH*68)
    print(f'{"Metric":<12} {"Trivial Pos":>14} {"Trivial Neg":>14} {"A1":>12} {"A2":>12}')
    print(sH*68)
    print(f'{"EC(λ)":<12} {ec_pos:>14.5f} {ec_neg:>14.5f} {ec1:>12.5f} {ec2:>12.5f}')
    print(f'{"FPR":<12} {1.0:>14.5f} {0.0:>14.5f} {fpr1:>12.5f} {fpr2:>12.5f}')
    print(f'{"FNR":<12} {0.0:>14.5f} {1.0:>14.5f} {fnr1:>12.5f} {fnr2:>12.5f}')
    print()
    print(f'A1 vs. Trivial Pos:  ΔEC(λ) = {ec1 - ec_pos:+.5f}  ({"beats" if ec1 < ec_pos else "does not beat"} Trivial Pos)')
    print(f'A1 vs. Trivial Neg:  ΔEC(λ) = {ec1 - ec_neg:+.5f}  ({"beats" if ec1 < ec_neg else "does not beat"} Trivial Neg)')
    print(f'A2 vs. Trivial Pos:  ΔEC(λ) = {ec2 - ec_pos:+.5f}  ({"beats" if ec2 < ec_pos else "does not beat"} Trivial Pos)')
    print(f'A2 vs. Trivial Neg:  ΔEC(λ) = {ec2 - ec_neg:+.5f}  ({"beats" if ec2 < ec_neg else "does not beat"} Trivial Neg)')
    print(sH*68)


def main():
    a1_data = dict(np.load(A1_RESULTS_PATH))
    a2_data = dict(np.load(A2_RESULTS_PATH))

    for cost_ratio in COST_RATIOS:
        print(dH*100)
        compare_EC(cost_ratio, a1_data, a2_data)
        print()
        compare_trivial(cost_ratio, a1_data, a2_data)
        print()
    print(dH*100)


if __name__ == '__main__':
    main()
