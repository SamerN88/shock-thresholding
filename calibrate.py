import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim

from preprocess_data import load_data_splits
from model import Ecg1LeadCNN
from config import FINAL_MODEL_PATH, CALIBRATED_DIR, CALIBRATED_MODEL_PATH


def calibrate(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'[Using device "{device}"]\n')

    # Load trained model (weights frozen; only T is learned)
    model = Ecg1LeadCNN().to(device)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f'Calibrating model from:  {FINAL_MODEL_PATH}\n')

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

    # Compute NLL loss before calibration
    # (NOTE: BCE is NLL for binary classification; NLL generalizes to multi-class)
    nll = nn.BCEWithLogitsLoss()
    nll_before = nll(all_logits, all_labels).item()

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

    # Compute NLL loss after calibration
    T = temperature.item()
    nll_after = nll(all_logits / T, all_labels).item()

    print(f'Temperature (T):         {T:.5f}')
    print(f'NLL before calibration:  {nll_before:.5f}')
    print(f'NLL after calibration:   {nll_after:.5f}')

    # Save model state & temperature together so a single load gives everything needed for inference
    Path(CALIBRATED_DIR).mkdir(parents=True, exist_ok=True)
    pt_path = Path(CALIBRATED_MODEL_PATH)
    torch.save({'model_state_dict': model.state_dict(), 'temperature': T}, pt_path)

    # Copy the training metadata JSON and append the temperature
    meta = json.loads(Path(FINAL_MODEL_PATH).with_suffix('.json').read_text())
    meta['temperature'] = T
    pt_path.with_suffix('.json').write_text(json.dumps(meta, indent=4))

    print(f'\nCalibrated model saved to:  {CALIBRATED_MODEL_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temperature-scale the trained model on the validation set')
    parser.add_argument('--device', type=str, default=None, help='Device to use, e.g. "cuda", "mps", "cpu" (default: auto-detect)')
    args = parser.parse_args()

    calibrate(device=args.device)
