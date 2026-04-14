import os
import shutil

from train import train
from calibrate import calibrate
from eval import evaluate
from config import (
    RESET_RANDOM_STATE,
    MODEL_DIR,
    FINAL_MODEL_PATH,
    CALIBRATED_DIR,
    CALIBRATED_MODEL_PATH,
    COST_RATIOS
)


# Convenient orchestration script to run the entire pipeline for Approach 1 (cost-sensitive thresholding)
def main():
    RESET_RANDOM_STATE()

    print('=' * 70)
    print('APPROACH 1: COST-SENSITIVE THRESHOLDING'.center(70))
    print('=' * 70)
    print(f'  - Train 1 model with pos_weight=1.0')
    print(f'  - Calibrate via temperature scaling')
    print(f"  - Evaluate at Elkan's optimal threshold θ*(λ) = 1/(λ+1) with")
    print(f'        λ = {", ".join(map(str, COST_RATIOS))}')
    print()
    print(f'Saving calibrated model to:  {CALIBRATED_DIR + os.path.sep}')
    print('=' * 70)
    print('\n\n')

    # Train model with symmetric costs (pos_weight=1); actual cost ratio will be encoded in the thresholding later
    ok = train(pos_weight=1)
    if not ok:
        print()
        print('=' * 100)
        print(f'Error while training Approach 1 model. Exiting.')
        return
    print()

    # Calibrate model so we can treat model output as a posterior probability (necessary for Elkan's thresholding)
    calibrate(model_path=FINAL_MODEL_PATH)

    # Delete model/ directory (final weights already moved to CALIBRATED_MODEL_PATH)
    shutil.rmtree(MODEL_DIR)

    print('\n\n')
    print('='*100)
    print('APPROACH 1 EVALUATION'.center(100))
    print(f'pos_weight = 1.0,  threshold = 1/(λ+1),  Calibrated'.center(100))
    print('=' * 100)
    print()

    # Evaluate calibrated model with each cost ratio; use Elkan's optimal threshold θ*(λ) = 1/(λ+1)
    for cost_ratio in COST_RATIOS:
        print(f'COST RATIO:  λ = {cost_ratio}\n')
        opt_threshold = 1 / (cost_ratio + 1)
        evaluate(model_path=CALIBRATED_MODEL_PATH, threshold=opt_threshold, cost_ratio=cost_ratio)
        print('\n' + '='*100 + '\n')


if __name__ == '__main__':
    main()
