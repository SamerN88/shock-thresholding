import os
import shutil
from pathlib import Path

import numpy as np

from train import train
from calibrate import calibrate
from eval import evaluate
from util import Glyphs, elkan_optimal_threshold
from config import (
    RESET_RANDOM_STATE,
    MODEL_DIR,
    FINAL_MODEL_PATH,
    CALIBRATED_DIR,
    CALIBRATED_MODEL_PATH,
    RESULTS_DIR,
    A1_RESULTS_PATH,
    COST_RATIOS
)


# Special characters, just for pretty display
dV = Glyphs.dV
dH = Glyphs.dH
dUL = Glyphs.dUL
dUR = Glyphs.dUR
dDL = Glyphs.dDL
dDR = Glyphs.dDR
d3U = Glyphs.d3U
d3D = Glyphs.d3D
d3L = Glyphs.d3L
d3R = Glyphs.d3R
d4 = Glyphs.d4
bul = Glyphs.bul


# Convenient orchestration script to run the entire pipeline for Approach 1 (cost-sensitive thresholding)
def main():
    RESET_RANDOM_STATE()

    print(dDR + dH*70 + dDL)
    print(dV + 'APPROACH 1: COST-SENSITIVE THRESHOLDING'.center(70) + dV)
    print(d3R + dH * 70 + d3L)
    print(dV + f'   {bul} Train 1 model with pos_weight=1.0'.ljust(70) + dV)
    print(dV + f'   {bul} Calibrate via temperature scaling'.ljust(70) + dV)
    print(dV + f"   {bul} Evaluate at Elkan's optimal threshold θ*(λ) = 1/(λ+1) for".ljust(70) + dV)
    print(dV + f'        λ = {", ".join(map(str, COST_RATIOS))}'.ljust(70) + dV)
    print(dV + ' '*70 + dV)
    print(dV + f' Saving calibrated model to:  {CALIBRATED_DIR + os.path.sep}'.ljust(70) + dV)
    print(dUR + dH*70 + dUL)
    print('\n\n')

    # Train model with symmetric costs (pos_weight=1); actual cost ratio will be encoded in the thresholding later
    ok = train(pos_weight=1)
    if not ok:
        print()
        print(dH*70)
        print(f'Error while training Approach 1 model. Exiting.')
        return
    print()

    # Calibrate model so we can treat model output as a posterior probability (necessary for Elkan's thresholding)
    calibrate(model_path=FINAL_MODEL_PATH)

    # Delete model/ directory (final weights already moved to CALIBRATED_MODEL_PATH)
    shutil.rmtree(MODEL_DIR)

    print('\n\n')
    print(dDR + dH*98 + dDL)
    print(dV + 'APPROACH 1 EVALUATION'.center(98) + dV)
    print(dV + f'pos_weight = 1.0,  threshold = 1/(λ+1),  Calibrated'.center(98) + dV)
    print(dUR + dH*98 + dUL)
    print()

    # Evaluate calibrated model with each cost ratio; use Elkan's optimal threshold θ*(λ) = 1/(λ+1)
    all_preds = {}
    labels = None
    for cost_ratio in COST_RATIOS:
        print(f'COST RATIO:  λ = {cost_ratio}\n')
        preds, labels = evaluate(
            model_path=CALIBRATED_MODEL_PATH,
            cost_ratio=cost_ratio,
            threshold=elkan_optimal_threshold(cost_ratio),
            dataset='test'
        )
        all_preds[cost_ratio] = preds
        print('\n' + dH*100 + '\n')

    # Save predictions for statistical significance test in compare.py
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    np.savez(A1_RESULTS_PATH, labels=labels, **{f'preds_{lam}': preds for lam, preds in all_preds.items()})
    print(f'Saved A1 predictions to:  {A1_RESULTS_PATH}')


if __name__ == '__main__':
    main()
