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
from util import Stopwatch, fmt_sec, flint
from config import (
    RANDOM_SEED,
    BATCH_SIZE,
    INIT_LEARNING_RATE,
    EPOCHS,
    LR_SCHEDULE_FACTOR,
    LR_SCHEDULE_PATIENCE,
    MODEL_DIR,
    CHECKPOINTS_DIR,
    FINAL_MODEL_PATH
)


def train(cost_ratio=None, resume=False, batch_size=None, lr=None, epochs=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'[Using device "{device}"]\n')

    explicit_lr = lr  # None if user didn't pass --lr explicitly; used for LR override on resume

    # Reconcile training config: if resuming, load previous config as defaults; explicit CLI args override
    # checkpoint values except cost_ratio, which must not change mid-training
    if resume:
        prev = _peek_checkpoint_config()

        if cost_ratio is not None and float(cost_ratio) != float(prev['cost_ratio']):
            print('\n' + '*'*64)
            print('WARNING:')
            print(f'    --cost_ratio={flint(cost_ratio)} conflicts with the original λ.')
            print(f'    Changing λ mid-training invalidates the model.')
            print(f'    Using original:  λ = {flint(prev["cost_ratio"])}')
            print('*'*64 + '\n')
        cost_ratio = prev['cost_ratio']
        batch_size = batch_size if batch_size is not None else prev['batch_size']
        lr = lr if lr is not None else prev['init_lr']
        epochs = epochs if epochs is not None else prev['total_epochs']
    else:
        cost_ratio = float(cost_ratio) if cost_ratio is not None else 1.0
        batch_size = batch_size if batch_size is not None else BATCH_SIZE
        lr = lr if lr is not None else INIT_LEARNING_RATE
        epochs = epochs if epochs is not None else EPOCHS

    if epochs <= 0:
        raise ValueError('epochs must be positive')

    # Instantiate model, loss function, optimizer, and LR scheduler
    model = Ecg1LeadCNN().to(device)
    loss_fxn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(cost_ratio)).to(device))  # cost_ratio is the "λ" in our cost-sensitive framework
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULE_FACTOR, patience=LR_SCHEDULE_PATIENCE)

    # Optionally resume training from a previously saved run
    if resume:
        start_epoch = _load_latest_checkpoint(model, optimizer, scheduler)
        if start_epoch > epochs:
            print('\nTraining already complete. Nothing to do.')
            print(f'(Increase epochs above {start_epoch-1} to continue training.)')
            return
        if explicit_lr is not None:
            # Override the checkpoint's LR and reset the scheduler so it treats the new LR as its
            # starting point — without this, the old scheduler state could immediately reduce the
            # new LR if it thinks training has been on a plateau
            for param_group in optimizer.param_groups:
                param_group['lr'] = explicit_lr
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULE_FACTOR, patience=LR_SCHEDULE_PATIENCE)
            print(f'LR overridden to {explicit_lr}; LR scheduler reset.')
    else:
        if not _setup_fresh_run():
            return
        start_epoch = 1

    # Load training and validation sets (not test).
    # The RNG is created here and re-seeded each epoch (RANDOM_SEED + epoch) so that each epoch's
    # shuffle is deterministic and identical whether training is interrupted or not.
    shuffle_rng = torch.Generator()
    train_loader, valid_loader, _ = load_data_splits(batch_size=batch_size, rng=shuffle_rng)

    # Display training config (hyperparams, etc.) before starting
    lines = [
        (f'    batch_size = {batch_size}', ''),
        (f'    lr = {lr}', '(reduced on plateau)'),
        (f'    epochs = {epochs}', f'(starting from epoch {start_epoch})'),
        (f'    λ = {flint(cost_ratio)}', '(cost ratio C_FN / C_FP)')
    ]
    width = 4 + max(len(l[0]) for l in lines)
    print()
    print('-' * 50)
    print('Training config:')
    for hyperparam, note in lines:
        print(hyperparam.ljust(width) + note)
    print('-'*50)
    print()
    print(f'* Checkpoint saves every epoch to:  {CHECKPOINTS_DIR + os.path.sep}')
    print()

    sw = Stopwatch()
    sw.start()

    # "Session epoch" is the epoch number in this current run, while "epoch" is the epoch number for the entire
    # training process; if we interrupt training and then resume it later, "epoch" continues from where it left
    # off but "session epoch" resets to 1. We need the session epoch to accurately compute the ETA.
    avg_epoch_runtime = 0
    for session_epoch, epoch in enumerate(range(start_epoch, epochs+1), 1):
        # Re-seed per epoch so each epoch's shuffle and any stochastic ops (e.g. dropout) are identical
        # whether training is uninterrupted or resumed from a checkpoint.
        shuffle_rng.manual_seed(RANDOM_SEED + epoch)
        torch.manual_seed(RANDOM_SEED + epoch)

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

        # Mean loss over batches (same scale as valid_loss)
        train_loss /= len(train_loader)

        # Validation loop (compute loss, accuracy, FPR, FNR, and MC(λ) on validation set)
        model.eval()
        valid_loss, acc, fpr, fnr = _validate(
            model=model,
            loss_fxn=loss_fxn,
            data_loader=valid_loader,
            device=device,
            desc=f'Epoch {epoch}/{epochs} [valid]'
        )
        MC_lam = fpr + cost_ratio*fnr  # misclassification cost ("clinical cost") w.r.t. cost ratio λ

        # Compute and display runtime info, and step scheduler
        epoch_runtime = sw.elapsed()  # includes both train and validation time
        avg_epoch_runtime += (epoch_runtime - avg_epoch_runtime) / session_epoch  # online mean
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        ETA = avg_epoch_runtime * (epochs - epoch)
        print(f'Epoch {f"{epoch}:".ljust(4)}    train_loss={train_loss:.5f}  valid_loss={valid_loss:.5f}  MC(λ)={MC_lam:.5f}  FPR={fpr:.5f}  FNR={fnr:.5f}  acc={acc:.5f}  lr={current_lr}  epoch_runtime={fmt_sec(epoch_runtime)}  ETA={fmt_sec(ETA)}')

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
            'valid_loss': valid_loss,
            'cost_ratio': cost_ratio,
            'mc_lam': MC_lam,
            'fpr': fpr,
            'fnr': fnr,
            'acc': acc
        }

        # Save a checkpoint (model and training state) each epoch
        _save_checkpoint(epoch, model, optimizer, scheduler, meta)

    # When training ends, save the final model and some info
    _save_final_model(model, meta)

    print()
    print('-'*50)
    print('Final stats:')
    print(f'    train_loss = {train_loss:.5f}')
    print(f'    valid_loss = {valid_loss:.5f}')
    print(f'    MC(λ)      = {MC_lam:.5f}')
    print(f'    FPR        = {fpr:.5f}')
    print(f'    FNR        = {fnr:.5f}')
    print(f'    acc        = {acc:.5f}')
    print()
    print(f'Total runtime: {fmt_sec(sw.total_elapsed())}')
    print('-' * 50)
    print()


def _validate(model, loss_fxn, data_loader, device, desc=None):
    """One pass over data_loader; returns (valid_loss, acc, FPR, FNR)."""
    valid_loss = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        # Validation loop
        for segs, labels in tqdm(data_loader, desc=desc, leave=False):
            segs, labels = segs.to(device), labels.to(device)
            logits = model(segs)
            valid_loss += loss_fxn(logits.squeeze(-1), labels.float()).item()
            preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy()  # 0.5 is training-time diagnostic only; final eval uses θ*(λ)
            labels = labels.cpu().numpy()
            tp += ((preds == 1) & (labels == 1)).sum()
            fp += ((preds == 1) & (labels == 0)).sum()
            tn += ((preds == 0) & (labels == 0)).sum()
            fn += ((preds == 0) & (labels == 1)).sum()
    valid_loss = valid_loss / len(data_loader)  # mean loss over batches
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return valid_loss, acc, fpr, fnr


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


def _peek_checkpoint_config():
    """Read config from the latest checkpoint's JSON without loading model states."""
    checkpoints = list(Path(CHECKPOINTS_DIR).glob('checkpoint-*.pt'))
    if not checkpoints:
        raise FileNotFoundError(f'resume=True but no checkpoint .pt files found in {CHECKPOINTS_DIR + os.path.sep}')
    latest = max(checkpoints, key=lambda p: int(p.stem.split('-')[1]))
    return json.loads(latest.with_suffix('.json').read_text())


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
    parser.add_argument('--resume',     action='store_true',    default=False,  help='Resume from latest checkpoint (default: False)')
    parser.add_argument('--batch_size', type=int,               default=None,   help=f'Batch size (default: {BATCH_SIZE}, or previous run value when resuming)')
    parser.add_argument('--lr',         type=float,             default=None,   help=f'Initial learning rate (default: {INIT_LEARNING_RATE}, or previous run value when resuming)')
    parser.add_argument('--epochs',     type=int,               default=None,   help=f'Total training epochs (default: {EPOCHS}, or previous run value when resuming)')
    parser.add_argument('--device',     type=str,               default=None,   help='Device to train on, e.g. "cuda", "mps", "cpu" (default: auto-detect)')
    parser.add_argument('--cost_ratio', type=float,             default=None,   help='Cost ratio λ = C_FN / C_FP (default: 1.0, or previous run value when resuming; cannot change mid-training)')
    args = parser.parse_args()

    try:
        train(cost_ratio=args.cost_ratio, resume=args.resume, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, device=args.device)
    except KeyboardInterrupt:
        print('\n\nTraining interrupted. Resume with --resume flag.')
