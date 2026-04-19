import time

import numpy as np


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.last_lap = None

    def start(self):
        self.start_time = self.last_lap = time.time()

    def lap(self):
        now = time.time()
        elapsed = now - self.last_lap
        self.last_lap = now
        return elapsed

    def elapsed(self):
        return time.time() - self.last_lap

    def total_elapsed(self):
        return time.time() - self.start_time


def fmt_sec(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.3f}s"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    secs = seconds % 60

    if seconds < 3600:
        return f"{minutes}m{int(secs)}s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours}h{minutes}m{secs}s"


def metrics(preds, labels):
    """
    Given binary predictions and labels, computes accuracy, true/false positives/negatives, FPR, and FNR.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    acc = (preds == labels).mean()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return acc, tp, fp, tn, fn, fpr, fnr


def EC(preds, labels, cost_ratio):
    """
    Computes the primary evaluation metric

        EC(λ)  =  P(Y=0)·FPR + λ·P(Y=1)·FNR  =  (FP + λ·FN) / N

    EC(λ) is the expected cost per example (under our data's class distribution) given a
    certain cost ratio λ. This is the main metric we use for comparing A1 vs. A2. It's
    easy to see that this quantity is an expectation; first define:

        - P(FP) = FP/N,  C_FP = 1
        - P(FN) = FN/N,  C_FN = λ

    where C_FP is the cost of a FP, and P(FP) is the probability of getting a FP on a
    randomly drawn example from our dataset; likewise for FN (FP = inappropriate shock,
    FN = missed shock). Thus, classifying a randomly drawn example from this dataset has
    an expected cost of

        E[cost] = C_FP·P(FP) + C_FN·P(FN)
                = 1·FP/N + λ·FN/N

    Hence we write EC(λ) = (FP + λ·FN) / N. Note that EC(λ) is the average misclassification
    cost per example, which both A1 and A2 are theoretically designed to minimize.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    n = len(labels)
    return (fp + cost_ratio * fn) / n


def ece(probs, labels, n_bins):
    """
    Expected Calibration Error: weighted average of |accuracy - confidence| per bin.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    result = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()  # bin confidence
        bin_acc = labels[mask].mean()  # bin accuracy
        result += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return result


def elkan_optimal_threshold(cost_ratio):
    """
    Elkan's optimal threshold p* = c10 / (c10 + c01), or equivalently in our notation,
    θ*(λ) = 1/(λ+1) where λ = c01 / c10
    """
    return 1 / (cost_ratio + 1)


def sigma_star(p, cost_ratio):
    """
    Optimal confidence output of a cost-sensitive model trained with pos_weight=λ (Approach 2).

        σ*(p) = λp / (1 − p + λp)

    This is the posterior P(Y=1|x) that a Bayes-optimal model trained under cost ratio λ would
    output for a true class-1 probability of p. Setting σ*(p) = 0.5 recovers Elkan's threshold:
    p = θ*(λ) = 1/(λ+1), establishing theoretical equivalence of A1 and A2.
    """
    return cost_ratio * p / (1 - p + cost_ratio * p)


class Glyphs:
    sV = chr(0x2502)   # │  single bar, vertical
    sH = chr(0x2500)   # ─  single bar, horizontal
    sUL = chr(0x2518)  # ┘  single bar, up left
    sUR = chr(0x2514)  # └  single bar, up right
    sDL = chr(0x2510)  # ┐  single bar, down left
    sDR = chr(0x250C)  # ┌  single bar, down right
    s3U = chr(0x2534)  # ┴  single bar, 3-way intersection pointing up
    s3D = chr(0x252C)  # ┬  single bar, 3-way intersection pointing down
    s3L = chr(0x2524)  # ┤  single bar, 3-way intersection pointing left
    s3R = chr(0x251C)  # ├  single bar, 3-way intersection pointing right
    s4 = chr(0x253C)   # ┼  single bar, 4-way intersection
    dV = chr(0x2551)   # ║  double bar, vertical
    dH = chr(0x2550)   # ═  double bar, horizontal
    dUL = chr(0x255D)  # ╝  double bar, up left
    dUR = chr(0x255A)  # ╚  double bar, up right
    dDL = chr(0x2557)  # ╗  double bar, down left
    dDR = chr(0x2554)  # ╔  double bar, down right
    d3U = chr(0x2569)  # ╩  double bar, 3-way intersection pointing up
    d3D = chr(0x2566)  # ╦  double bar, 3-way intersection pointing down
    d3L = chr(0x2563)  # ╣  double bar, 3-way intersection pointing left
    d3R = chr(0x2560)  # ╠  double bar, 3-way intersection pointing right
    d4 = chr(0x256C)   # ╬  double bar, 4-way intersection
    bul = chr(0x2022)  # •  bullet
