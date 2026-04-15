import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from preprocess_data import load_data_splits
from model import Ecg1LeadCNN


def evaluate(model_path, *, cost_ratio, threshold=0.5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'[Using device "{device}"]\n')

    # Load model, pos_weight, and temperature
    model_path = Path(model_path)
    bundle = torch.load(model_path, map_location=device, weights_only=True)
    model = Ecg1LeadCNN().to(device)
    model.load_state_dict(bundle['model_state_dict'])
    model.eval()
    pos_weight = bundle['pos_weight']  # cost_ratio is the λ being evaluated; for A2 pos_weight=λ, for A1 pos_weight=1 always
    temperature = bundle['temperature']

    '''
    WARNING: Caller may pass in values that are incompatible with each other (examples below); we don't check 
    for all these cases, it's the caller's responsibility to provide sensible arguments.

        Some invalid cases:
            - pos_weight != 1, but cost_ratio != pos_weight
              PROBLEM: if pos_weight != 1 then it's A2, so cost_ratio must equal pos_weight
            - pos_weight != 1, but threshold != 0.5
              PROBLEM: if pos_weight != 1 then it's A2, so threshold must be 0.5
            - threshold != 0.5, but threshold != 1/(λ+1)
              PROBLEM: if threshold != 0.5 then it's A1, so threshold must equal Elkan's θ*(λ) = 1/(λ+1)

    There are others. This script does not differentiate between A1 and A2. It simply takes a model, some
    hyperparams, and evaluates on the test set. Again, it's the caller's job to provide sensible arguments.
    '''

    print(f'Model:        {model_path}')
    print(f'pos_weight:   {pos_weight}')
    print(f'temperature:  {temperature}')
    print(f'cost_ratio:   {cost_ratio}')
    print(f'threshold:    {threshold}')
    print()

    # We only evaluate on the test set (unseen data)
    _, _, test_loader = load_data_splits()

    loss_fxn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)).to(device))

    test_loss = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for segs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            segs, labels = segs.to(device), labels.to(device)
            logits = model(segs).squeeze(-1)
            test_loss += loss_fxn(logits, labels.float()).item()
            probs = torch.sigmoid(logits / temperature)
            preds = (probs >= threshold).long().cpu().numpy()
            labels = labels.cpu().numpy()
            tp += ((preds == 1) & (labels == 1)).sum()
            fp += ((preds == 1) & (labels == 0)).sum()
            tn += ((preds == 0) & (labels == 0)).sum()
            fn += ((preds == 0) & (labels == 1)).sum()

    # Compute test metrics
    test_loss /= len(test_loader)
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mc_lam = fpr + cost_ratio * fnr  # misclassification cost w.r.t. λ; most important metric for comparing A1 and A2

    print('-'*70)
    print('Test metrics:')
    print(f'    test_loss = {test_loss:.5f}')
    print(f'    MC(λ)     = {mc_lam:.5f}')
    print(f'    FPR       = {fpr:.5f}')
    print(f'    FNR       = {fnr:.5f}')
    print(f'    acc       = {acc:.5f}')
    print(f'    TP={tp}  FP={fp}  TN={tn}  FN={fn}')
    print('-'*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the test set')
    parser.add_argument('--model_path', type=str,   required=True,  help='Path to the .pt model file')
    parser.add_argument('--cost_ratio', type=float, required=True,  help='Cost ratio λ for MC(λ) evaluation')
    parser.add_argument('--threshold',  type=float, default=0.5,    help='Decision threshold θ (default: 0.5; use 1/(λ+1) for A1 cost-sensitive eval)')
    parser.add_argument('--device',     type=str,   default=None,   help='Device to use, e.g. "cuda", "mps", "cpu" (default: auto-detect)')
    args = parser.parse_args()

    evaluate(model_path=args.model_path, cost_ratio=args.cost_ratio, threshold=args.threshold, device=args.device)
