import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from preprocess_data import load_data_splits
from model import Ecg1LeadCNN
from config import CALIBRATED_DIR, CALIBRATED_MODEL_PATH


def calibrate(model_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'[Using device "{device}"]\n')

    # Load trained model (weights frozen; only T is learned)
    model_path = Path(model_path)
    model_bundle = torch.load(model_path, map_location=device, weights_only=True)
    model = Ecg1LeadCNN().to(device)
    model.load_state_dict(model_bundle['model_state_dict'])
    model.eval()

    print(f'Calibrating model from:  {model_path}')
    print('(using validation set)\n')

    # Load validation set only
    _, valid_loader, _ = load_data_splits()

    # Collect all validation logits and labels
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for segs, labels in valid_loader:
            segs = segs.to(device)  # only move segments to model's device; labels don't need to
            logits = model(segs)
            all_logits.extend(logits.cpu())
            all_labels.extend(labels)
    all_logits = torch.tensor(all_logits).squeeze(-1)
    all_labels = torch.tensor(all_labels).float()

    # Use NLL as our loss function (BCE is NLL for binary classification; NLL generalizes to multi-class)
    nll = nn.BCEWithLogitsLoss()

    # Compute NLL and ECE before calibration
    probs_before = torch.sigmoid(all_logits)
    nll_before = nll(all_logits, all_labels).item()
    ece_before = compute_ece(probs_before, all_labels, n_bins=10)

    # Learn T by minimizing NLL on the validation logits (Guo et al. 2017)
    # Logits and temperature stay on CPU since this is a small 1D convex optimization
    temperature = nn.Parameter(torch.ones(1))
    optimizer = optim.LBFGS([temperature], lr=0.1, max_iter=1000, line_search_fn='strong_wolfe')  # L-BFGS is an efficient convex optimizer

    # This is the function we minimize when we run L-BFGS below (with temperature as the parameter)
    def objective():
        optimizer.zero_grad()
        loss = nll(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    # Full L-BFGS run up to max_iter=1000 (NOT a single L-BFGS step)
    optimizer.step(objective)
    T = temperature.item()  # final optimized T

    # Compute NLL and ECE after calibration
    probs_after = torch.sigmoid(all_logits / T)
    nll_after = nll(all_logits / T, all_labels).item()
    ece_after = compute_ece(probs_after, all_labels, n_bins=10)

    print(f'Done.\nTemperature (T):  {T:.5f}\n')
    print('Before calibration:')
    print(f'    NLL:  {nll_before:.5f}')
    print(f'    ECE:  {ece_before:.5f}')
    print('After calibration:')
    print(f'    NLL:  {nll_after:.5f}')
    print(f'    ECE:  {ece_after:.5f}')

    # Save model state & temperature together so a single load gives everything needed for inference
    Path(CALIBRATED_DIR).mkdir(parents=True, exist_ok=True)
    pt_path = Path(CALIBRATED_MODEL_PATH)
    model_bundle['temperature'] = T
    torch.save(model_bundle, pt_path)

    # Copy the training info JSON and append the temperature
    info = json.loads(model_path.with_suffix('.json').read_text())
    info['temperature'] = T
    pt_path.with_suffix('.json').write_text(json.dumps(info, indent=4))

    print(f'\nCalibrated model saved to:  {CALIBRATED_MODEL_PATH}')


def compute_ece(probs, labels, n_bins):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (
            (probs >= bins[i]) & (probs < bins[i + 1])
            if i < n_bins - 1 else
            (probs >= bins[i]) & (probs <= bins[i + 1])
        )
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()  # bin confidence
        bin_acc = labels[mask].mean()  # bin accuracy
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)

    return ece


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temperature-scale the trained model on the validation set')
    parser.add_argument('--model_path', type=str,   required=True,  help='Path to the .pt model file to calibrate')
    parser.add_argument('--device',     type=str,   default=None,   help='Device to use, e.g. "cuda", "mps", "cpu" (default: auto-detect)')
    args = parser.parse_args()

    calibrate(model_path=args.model_path, device=args.device)
