import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from preprocess_data import load_data_splits
from model import Ecg1LeadCNN
from util import metrics, EC


def evaluate(model_path, *, cost_ratio, threshold=0.5, dataset="test", device=None):
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
    hyperparams, and evaluates on the dataset. Again, it's the caller's job to provide sensible arguments.
    '''

    # By default we evaluate on the test set (unseen data), but the user can choose some other set ("train" or "valid")
    train_loader, valid_loader, test_loader = load_data_splits()
    if dataset == 'train':
        data_loader = train_loader
    elif dataset == 'valid':
        data_loader = valid_loader
    elif dataset == 'test':
        data_loader = test_loader
    else:
        raise ValueError(f'invalid dataset name; expected "train", "valid", or "test", got "{dataset}"')

    print(f'Model:        {model_path}')
    print(f'Dataset:      "{dataset}"')
    print(f'pos_weight:   {pos_weight}')
    print(f'temperature:  {temperature}')
    print(f'cost_ratio:   {cost_ratio}')
    print(f'threshold:    {threshold}')
    print()

    loss_fxn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)).to(device))

    # Run inference
    loss = 0
    all_logits = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for segs, labels in tqdm(data_loader, desc='Evaluating', leave=False):
            segs = segs.to(device)
            labels = labels.to(device)

            logits = model(segs).squeeze(-1)
            probs = torch.sigmoid(logits / temperature)
            preds = (probs >= threshold).long()

            loss += loss_fxn(logits, labels.float()).item()

            all_logits.extend(logits.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute performance metrics
    loss /= len(data_loader)
    acc, tp, fp, tn, fn, fpr, fnr = metrics(all_preds, all_labels)
    ec_lam = EC(all_preds, all_labels, cost_ratio)

    print('-'*70)
    print(f'Performance metrics (dataset: "{dataset}"):')
    print(f'    loss  = {loss:.5f}')
    print(f'    EC(λ) = {ec_lam:.5f}')
    print(f'    FPR   = {fpr:.5f}')
    print(f'    FNR   = {fnr:.5f}')
    print(f'    acc   = {acc:.5f}')
    print(f'    TP={tp}  FP={fp}  TN={tn}  FN={fn}')
    print('-'*70)

    return (
        np.array(all_logits),
        np.array(all_preds),
        np.array(all_labels)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on a dataset')
    parser.add_argument('--model_path', type=str,   required=True,  help='Path to the .pt model file')
    parser.add_argument('--cost_ratio', type=float, required=True,  help='Cost ratio λ for EC(λ) evaluation')
    parser.add_argument('--threshold',  type=float, default=0.5,    help='Decision threshold θ; use 1/(λ+1) for A1 cost-sensitive eval (default: 0.5)')
    parser.add_argument('--dataset',    type=str,   default="test", help='Dataset to evaluate on; "train", "valid", or "test" (default: "test")')
    parser.add_argument('--device',     type=str,   default=None,   help='Device to use, e.g. "cuda", "mps", "cpu" (default: auto-detect)')
    args = parser.parse_args()

    evaluate(model_path=args.model_path, cost_ratio=args.cost_ratio, threshold=args.threshold, dataset=args.dataset, device=args.device)
