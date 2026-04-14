import os
import shutil
from pathlib import Path

from train import train
from config import MODEL_DIR, FINAL_MODEL_PATH, COST_SENSITIVE_DIR


def main():
    # TODO: replace these dummy values with clinically-grounded cost ratios
    cost_ratios = [1, 2, 5, 10, 100, 1000]

    cs_dir_path = Path(COST_SENSITIVE_DIR)
    cs_dir_path.mkdir(parents=True, exist_ok=True)

    print('+'*50)
    print(f'Training {len(cost_ratios)} models with:')
    print(f'    λ = {", ".join(map(str, cost_ratios))}')
    print(f'\nSaving all models to:  {COST_SENSITIVE_DIR + os.path.sep}')
    print('+'*50)
    print('\n')
    for cost_ratio in cost_ratios:
        # Train a new model using an explicit cost ratio
        ok = train(cost_ratio=cost_ratio)
        if not ok:
            print()
            print('+'*100)
            print(f'Error while training λ={cost_ratio}. Exiting.')
            return

        # Move the trained model and metadata to a dedicated directory
        cs_model_dir = os.path.join(COST_SENSITIVE_DIR, f'lam-{cost_ratio}')
        Path(cs_model_dir).mkdir(parents=True, exist_ok=True)
        cs_model_path = Path(os.path.join(cs_model_dir, f'lam-{cost_ratio}.pt'))
        shutil.move(FINAL_MODEL_PATH, cs_model_path)
        shutil.move(Path(FINAL_MODEL_PATH).with_suffix('.json'), cs_model_path.with_suffix('.json'))

        # Delete contents of model/ to prepare for next run
        shutil.rmtree(MODEL_DIR)

        print('\n' + '+'*100 + '\n')


if __name__ == '__main__':
    main()
