import os
import shutil
import argparse
from pathlib import Path

import numpy as np

from train import train
from eval import evaluate
from util import Glyphs
from config import (
    RESET_RANDOM_STATE,
    MODEL_DIR,
    FINAL_MODEL_PATH,
    COST_SENSITIVE_DIR,
    RESULTS_DIR,
    A2_RESULTS_PATH,
    COST_RATIOS
)


# Special characters, just for pretty display
dV = Glyphs.dV
dH = Glyphs.dH
dUL = Glyphs.dUL
dUR = Glyphs.dUR
dDL = Glyphs.dDL
dDR = Glyphs.dDR
d3L = Glyphs.d3L
d3R = Glyphs.d3R
bul = Glyphs.bul


# Convenient orchestration script to run the entire pipeline for Approach 2 (cost-sensitive training)
def main(yes=False):
    RESET_RANDOM_STATE()
    print(dDR + dH*70 + dDL)
    print(dV + 'APPROACH 2: COST-SENSITIVE TRAINING'.center(70) + dV)
    print(d3R + dH*70 + d3L)
    print(dV + f'   {bul} Train {len(COST_RATIOS)} cost-sensitive models with'.ljust(70) + dV)
    print(dV + f'        pos_weight = λ = {", ".join(map(str, COST_RATIOS))}'.ljust(70) + dV)
    print(dV + f'   {bul} Evaluate each model at default threshold θ=0.5 (no calibration)'.ljust(70) + dV)
    print(dV + ' '*70 + dV)
    print(dV + f' Saving all models to:  {COST_SENSITIVE_DIR + os.path.sep}'.ljust(70) + dV)
    print(dUR + dH*70 + dUL)
    print('\n\n')

    # Ensure directory for storing λ-aware models exists
    Path(COST_SENSITIVE_DIR).mkdir(parents=True, exist_ok=True)

    # Train one model per λ; cost is encoded in training, via cost-weighted BCE loss (pos_weight=λ)
    for cost_ratio in COST_RATIOS:
        ok = train(pos_weight=cost_ratio, yes=yes)
        if not ok:
            print()
            print(dH*70)
            print(f'Error while training Approach 2 model (λ={cost_ratio}). Exiting.')
            return

        # Move the trained model and training info to a dedicated directory
        cs_model_dir = os.path.join(COST_SENSITIVE_DIR, f'lam-{cost_ratio}')
        Path(cs_model_dir).mkdir(parents=True, exist_ok=True)
        cs_model_path = Path(os.path.join(cs_model_dir, f'lam-{cost_ratio}.pt'))
        shutil.move(FINAL_MODEL_PATH, cs_model_path)
        shutil.move(Path(FINAL_MODEL_PATH).with_suffix('.json'), cs_model_path.with_suffix('.json'))

        # Delete model/ directory to prepare for the next run (final weights already moved to COST_SENSITIVE_DIR)
        shutil.rmtree(MODEL_DIR)

        print(f'\nMoved model to:  {cs_model_dir + os.path.sep}')
        print('\n' + dH*100 + '\n')

    print('\n')
    print(dDR + dH*98 + dDL)
    print(dV + 'APPROACH 2 EVALUATION'.center(98) + dV)
    print(dV + f'pos_weight = λ,  threshold = 0.5,  Not calibrated'.center(98) + dV)
    print(dUR + dH*98 + dUL)
    print()

    # Evaluate each λ-aware model at default threshold θ=0.5
    all_preds = {}
    for cost_ratio in COST_RATIOS:
        print(f'COST RATIO:  λ = {cost_ratio}\n')
        model_path = os.path.join(COST_SENSITIVE_DIR, f'lam-{cost_ratio}', f'lam-{cost_ratio}.pt')
        _, preds, labels = evaluate(
            model_path=model_path,
            cost_ratio=cost_ratio,
            threshold=0.5,
            dataset='test'
        )
        all_preds[cost_ratio] = preds
        print('\n' + dH*100 + '\n')

    # Save predictions for statistical significance test in compare.py
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    np.savez(
        A2_RESULTS_PATH,
        labels=labels,
        **{f'preds_{lam}': preds for lam, preds in all_preds.items()}
    )
    print(f'Saved A2 predictions to:  {A2_RESULTS_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true', default=False, help='Skip confirmation prompt when overwriting existing model directory')
    args = parser.parse_args()
    main(yes=args.yes)
