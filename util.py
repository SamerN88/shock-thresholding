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


def MC(preds, labels, cost_ratio):
    """
    Computes the core performance metric MC(λ) = FPR + λ*FNR (misclassification cost w.r.t. cost ratio λ).
    MC(λ) is the most important metric for comparing A1 and A2 as it estimates the "clinical cost".
    """
    *_, fpr, fnr = metrics(preds, labels)
    return fpr + cost_ratio * fnr


def elkan_optimal_threshold(cost_ratio):
    """
    Elkan's optimal threshold p* = c10 / (c10 + c01), or equivalently in our notation,
    θ*(λ) = 1/(λ+1) where λ = c01 / c10
    """
    return 1 / (cost_ratio + 1)


class Glyphs:
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
