import os

import torch


# RANDOM STATE (FOR REPRODUCIBILITY) -----------------------------------------------------------------------------------

# Fixed random seed -- all RNGs rely on this
RANDOM_SEED = 61

# Force deterministic CUDA kernels (trades some performance for reproducibility)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def RESET_RANDOM_STATE():
    """Re-seed RNGs to RANDOM_SEED. Call at the start of each training run."""
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

RESET_RANDOM_STATE()  # seed on import

# DATA PREPROCESSING CONFIG --------------------------------------------------------------------------------------------

VFDB_VALID_P = 0.2  # proportion of VFDB for validation split
WINDOW_SEC = 2  # ECG segment length for each example, in sec
MAX_NAN_GAP_SEC = 0.1*WINDOW_SEC  # for NaNs in ECG samples, linearly interpolate gaps up to a limit; drop larger gaps
NORMALIZE = True  # normalize ECG segments (per-segment)
SPLITS_NPZ_PATH = os.path.join('data', 'splits.npz')  # path for saving our data splits in a .npz file (using NumPy)

# TRAINING CONFIG ------------------------------------------------------------------------------------------------------

BATCH_SIZE = 64
INIT_LEARNING_RATE = 0.01
EPOCHS = 1
LR_SCHEDULE_FACTOR = 0.5  # multiply LR by this on plateau
LR_SCHEDULE_PATIENCE = 15  # number of epochs without valid_loss improvement before reducing LR

# MODEL PATHS ----------------------------------------------------------------------------------------------------------

MODEL_DIR = 'model'
CHECKPOINTS_DIR = os.path.join(MODEL_DIR, 'checkpoints')
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.pt')
CALIBRATED_DIR = 'calibrated'
CALIBRATED_MODEL_PATH = os.path.join(CALIBRATED_DIR, 'calibrated.pt')
COST_SENSITIVE_DIR = 'cost_sensitive'

# OTHER ----------------------------------------------------------------------------------------------------------------

# Labels
SHOCKABLE = 1
NON_SHOCKABLE = 0
COST_RATIOS = [1.0, 2.0, 5.0, 10.0, 100.0]  # TODO: replace these dummy values with clinically-grounded cost ratios
