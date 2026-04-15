import os
import shutil

from train import train
from calibrate import calibrate
from eval import evaluate
from util import Glyphs
from config import (
    RESET_RANDOM_STATE,
    MODEL_DIR,
    FINAL_MODEL_PATH,
    CALIBRATED_DIR,
    CALIBRATED_MODEL_PATH,
    COST_RATIOS
)


# Special characters, just for pretty display
dH = Glyphs.dH
dV = Glyphs.dV
dDR = Glyphs.dDR
dDL = Glyphs.dDL
dUR = Glyphs.dUR
dUL = Glyphs.dUL
dVR = Glyphs.dVR
dVL = Glyphs.dVL
dHD = Glyphs.dHD
dHU = Glyphs.dHU
d4 = Glyphs.d4
bul = Glyphs.bul


# Convenient orchestration script to run the entire pipeline for Approach 1 (cost-sensitive thresholding)
def main():
    RESET_RANDOM_STATE()

    print(dDR + dH*70 + dDL)
    print(dV + 'APPROACH 1: COST-SENSITIVE THRESHOLDING'.center(70) + dV)
    print(dVR + dH*70 + dVL)
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
    for cost_ratio in COST_RATIOS:
        print(f'COST RATIO:  λ = {cost_ratio}\n')
        opt_threshold = 1 / (cost_ratio + 1)
        evaluate(model_path=CALIBRATED_MODEL_PATH, threshold=opt_threshold, cost_ratio=cost_ratio, show_device=False)
        print('\n' + dH*100 + '\n')


if __name__ == '__main__':
    main()
