import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from preprocess_data import load_data_splits
from model import Ecg1LeadCNN


def evaluate(model_path, threshold=0.5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'[Using device "{device}"]\n')

    model_path = Path(model_path)
    bundle = torch.load(model_path, map_location=device, weights_only=True)
    model = Ecg1LeadCNN().to(device)
    model.load_state_dict(bundle['model_state_dict'])
    model.eval()
    cost_ratio = bundle['cost_ratio']
    temperature = bundle['temperature']

    print(f'Model:        {model_path}')
    print(f'cost_ratio:   {cost_ratio}')
    print(f'temperature:  {temperature}')
    print(f'threshold:    {threshold}')
    print()

    _, valid_loader, _ = load_data_splits()

    loss_fxn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(cost_ratio)).to(device))

    valid_loss = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for segs, labels in tqdm(valid_loader, desc='Evaluating', leave=False):
            segs, labels = segs.to(device), labels.to(device)
            logits = model(segs).squeeze(-1)
            valid_loss += loss_fxn(logits, labels.float()).item()
            probs = torch.sigmoid(logits / temperature)
            preds = (probs >= threshold).long().cpu().numpy()
            labels = labels.cpu().numpy()
            tp += ((preds == 1) & (labels == 1)).sum()
            fp += ((preds == 1) & (labels == 0)).sum()
            tn += ((preds == 0) & (labels == 0)).sum()
            fn += ((preds == 0) & (labels == 1)).sum()

    valid_loss /= len(valid_loader)
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mc_lam = fpr + cost_ratio * fnr

    print('-' * 50)
    print('Validation metrics:')
    print(f'    valid_loss = {valid_loss:.5f}')
    print(f'    MC(λ)      = {mc_lam:.5f}')
    print(f'    FPR        = {fpr:.5f}')
    print(f'    FNR        = {fnr:.5f}')
    print(f'    acc        = {acc:.5f}')
    print(f'    TP={tp}  FP={fp}  TN={tn}  FN={fn}')
    print('-' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the validation set')
    parser.add_argument('--model_path', type=str,   required=True,  help='Path to the .pt model file')
    parser.add_argument('--threshold',  type=float, default=0.5,    help='Decision threshold θ (default: 0.5; use 1/(1+λ) for A1 cost-sensitive eval)')
    parser.add_argument('--device',     type=str,   default=None,   help='Device to use, e.g. "cuda", "mps", "cpu" (default: auto-detect)')
    args = parser.parse_args()

    evaluate(model_path=args.model_path, threshold=args.threshold, device=args.device)
