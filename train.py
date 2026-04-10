import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from preprocess_data import load_data_splits
from model import Ecg1LeadCNN
from util import Stopwatch, fmt_sec
from config import (
    BATCH_SIZE,
    INIT_LEARNING_RATE,
    EPOCHS,
    LR_SCHEDULE_FACTOR,
    LR_SCHEDULE_PATIENCE,
    MODEL_DIR,
    CHECKPOINTS_DIR,
    FINAL_MODEL_PATH
)


def train(resume=False, batch_size=BATCH_SIZE, lr=INIT_LEARNING_RATE, epochs=EPOCHS, device=None):
    if epochs <= 0:
        raise ValueError('epochs must be positive')

    # Load training and validation sets (not test)
    train_loader, valid_loader, _ = load_data_splits(batch_size=batch_size)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'[Using device "{device}"]\n')

    # Instantiate model, loss function, optimizer, and LR scheduler
    model = Ecg1LeadCNN().to(device)
    loss_fxn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULE_FACTOR, patience=LR_SCHEDULE_PATIENCE)

    # Optionally resume training from a previously saved run
    if resume:
        start_epoch = _load_latest_checkpoint(model, optimizer, scheduler)
    else:
        if not _setup_fresh_run():
            return
        start_epoch = 1

    print()
    print('-' * 50)
    print('Training config:')
    print(f'    batch_size = {batch_size}')
    print(f'    lr = {lr}  (reduced on plateau)')
    print(f'    epochs = {epochs}  (starting from epoch {start_epoch})')
    print('-'*50)
    print()
    print(f'* Checkpoint saved every epoch to:  {CHECKPOINTS_DIR + os.path.sep}')
    print()

    sw = Stopwatch()
    sw.start()

    # "Session epoch" is the epoch number in this current run, while "epoch" is the epoch number for the entire
    # training process; if we interrupt training and then resume it later, "epoch" continues from where it left
    # off but "session epoch" resets to 1. We need the session epoch to accurately compute the ETA.
    avg_epoch_runtime = 0
    for session_epoch, epoch in enumerate(range(start_epoch, epochs+1), 1):
        sw.lap()
        model.train()

        # Training loop (run batches through model, compute loss, back-propagate)
        train_loss = 0
        for segs, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [train]', leave=False):
            segs, labels = segs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(segs)
            loss = loss_fxn(logits.squeeze(-1), labels.float())  # squeeze: (B,1) -> (B,); .float(): DataLoader yields int64, loss expects float32
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Mean loss over batches (same scale as val_loss)
        train_loss /= len(train_loader)

        # Validation loop (compute loss, accuracy, FPR, FNR on validation set)
        model.eval()
        val_loss, acc, fpr, fnr = _validate(
            model=model,
            loss_fxn=loss_fxn,
            data_loader=valid_loader,
            device=device,
            desc=f'Epoch {epoch}/{epochs} [val]  '
        )

        # Compute and display runtime info, and step scheduler
        epoch_runtime = sw.elapsed()  # includes both train and validation time
        avg_epoch_runtime += (epoch_runtime - avg_epoch_runtime) / session_epoch  # online mean
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        ETA = avg_epoch_runtime * (epochs - epoch)
        print(f'Epoch {(str(epoch) + ":").ljust(4)}    train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  acc={acc:.5f}  FPR={fpr:.5f}  FNR={fnr:.5f}  lr={current_lr}  epoch_runtime={fmt_sec(epoch_runtime)}  ETA={fmt_sec(ETA)}')

        # Info we save to a JSON for each epoch
        meta = {
            'epoch': epoch,
            'total_epochs': epochs,
            'batch_size': batch_size,
            'init_lr': lr,
            'current_lr': current_lr,
            'lr_schedule_factor': LR_SCHEDULE_FACTOR,
            'lr_schedule_patience': LR_SCHEDULE_PATIENCE,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'acc': acc,
            'fpr': fpr,
            'fnr': fnr,
        }

        # Skip checkpoint at final epoch; final model saved separately below
        if epoch < epochs:
            _save_checkpoint(epoch, model, optimizer, scheduler, meta)  # save the model at end of each epoch

    # Save the final model
    _save_final_model(model, meta)

    print()
    print('-'*50)
    print('Final stats:')
    print(f'    train_loss = {train_loss:.5f}')
    print(f'    valid_loss = {val_loss:.5f}')
    print(f'    acc        = {acc:.5f}')
    print(f'    FPR        = {fpr:.5f}')
    print(f'    FNR        = {fnr:.5f}')
    print()
    print(f'Total runtime: {fmt_sec(sw.total_elapsed())}')
    print('-' * 50)
    print()


def _validate(model, loss_fxn, data_loader, device, desc=None):
    """One pass over data_loader; returns (val_loss, acc, FPR, FNR)."""
    val_loss = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        # Validation loop
        for segs, labels in tqdm(data_loader, desc=desc, leave=False):
            segs, labels = segs.to(device), labels.to(device)
            logits = model(segs)
            val_loss += loss_fxn(logits.squeeze(-1), labels.float()).item()
            preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy()  # 0.5 is training-time diagnostic only; final eval uses θ*(λ)
            labels = labels.cpu().numpy()
            tp += ((preds == 1) & (labels == 1)).sum()
            fp += ((preds == 1) & (labels == 0)).sum()
            tn += ((preds == 0) & (labels == 0)).sum()
            fn += ((preds == 0) & (labels == 1)).sum()
    val_loss = val_loss / len(data_loader)  # mean loss over batches
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return val_loss, acc, fpr, fnr


def _setup_fresh_run():
    model_dir = Path(MODEL_DIR)
    if model_dir.exists() and any(p.is_file() for p in model_dir.rglob('*')):
        try:
            confirm = input(f'\nWARNING: {MODEL_DIR + os.path.sep} is non-empty. THIS WILL DELETE ALL EXISTING CHECKPOINTS AND MODELS.\nType "yes" to confirm: ')
        except KeyboardInterrupt:
            print('\n\nAborted.')
            exit()
        if confirm.strip().lower() != 'yes':
            print('\nAborted.')
            return False
        shutil.rmtree(model_dir)
    Path(CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    return True


def _save_checkpoint(epoch, model, optimizer, scheduler, meta):
    base = Path(CHECKPOINTS_DIR) / f'checkpoint-{epoch}'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, base.with_suffix('.pt'))
    base.with_suffix('.json').write_text(json.dumps(meta, indent=4))


def _load_latest_checkpoint(model, optimizer, scheduler):
    checkpoints = list(Path(CHECKPOINTS_DIR).glob('checkpoint-*.pt'))
    if not checkpoints:
        raise FileNotFoundError(f'resume=True but no checkpoint .pt files found in {CHECKPOINTS_DIR + os.path.sep}')

    latest = max(checkpoints, key=lambda p: int(p.stem.split('-')[1]))

    checkpoint = torch.load(latest, weights_only=False)  # checkpoint contains optimizer/scheduler state, not just tensors
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    print(f'\nResuming from:  {latest}  (epoch {checkpoint["epoch"]} -> continuing from epoch {start_epoch})')
    return start_epoch


def _save_final_model(model, meta):
    pt_path = Path(FINAL_MODEL_PATH)
    torch.save(model.state_dict(), pt_path)  # weights only; optimizer/scheduler state not needed post-training
    pt_path.with_suffix('.json').write_text(json.dumps(meta, indent=4))
    print(f'\nDone training. Final model saved:  {pt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ECG shock classification model')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from latest checkpoint')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=INIT_LEARNING_RATE, help=f'Initial learning rate (default: {INIT_LEARNING_RATE})')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--device', type=str, default=None, help='Device to train on, e.g. "cuda", "mps", "cpu" (default: auto-detect)')
    args = parser.parse_args()

    train(resume=args.resume, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, device=args.device)
