import os
import random

import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    RANDOM_SEED,
    SHOCKABLE,
    NON_SHOCKABLE,
    VFDB_VALID_P,
    WINDOW_SEC,
    BATCH_SIZE,
    MAX_NAN_GAP_SEC,
    NORMALIZE,
    SPLITS_NPZ_PATH
)


# CONSTANTS FOR DATASET ACCESS -----------------------------------------------------------------------------------------


# Define all DB names
VFDB = 'vfdb'
CUDB = 'cudb'

# Dataset paths
VFDB_DIR = os.path.join('data', 'vfdb', 'physionet.org', 'files', 'vfdb', '1.0.0')
CUDB_DIR = os.path.join('data', 'cudb', 'physionet.org', 'files', 'cudb', '1.0.0')

# Get DB path by name
DB_DIRS = {
    VFDB: VFDB_DIR,
    CUDB: CUDB_DIR
}


# HELPER FUNCTIONS FOR DATA RECORDS ------------------------------------------------------------------------------------


def dbfp(db, fn):
    return os.path.join(DB_DIRS[db], fn)


def get_record_names(db):
    with open(dbfp(db, "RECORDS"), 'r') as f:
        lines = f.readlines()
    return [l.strip() for l in lines if not (len(l) == 0 or l.isspace())]


def clean_aux_note(aux_note):
    return aux_note.strip().strip('(\0').strip()


# GENERATE EXAMPLES (NAN HANDLING, SEGMENTING, LABELING, NORMALIZING, ETC.) --------------------------------------------


# Just a potential convenience function, we don't have to use it
def get_examples_by_record(db, window_sec):
    return {
        VFDB: vfdb__get_examples_by_record,
        CUDB: cudb__get_examples_by_record
    }[db](window_sec)


def vfdb__get_examples_by_record(window_sec, normalize=True):
    """
    For VFDB, all annotations are "+" rhythm-change markers with an aux_note label (e.g. "(VT", "(N", etc.).
    Each annotation marks the start of a new rhythm interval; the interval ends at the next annotation.
    Labels are derived directly from the aux_note: shockable if VT/VF/VFL/VFIB, non-shockable otherwise.
    The signal before the first annotation is excluded (rhythm label ambiguous).
    The last interval runs from the final annotation to end-of-record.
    """

    def is_shockable(aux_note):
        rhythm = clean_aux_note(aux_note)
        return rhythm in {'VF', 'VFIB', 'VFL', 'VT'}

    def aux_note_to_label(aux_note):
        return SHOCKABLE if is_shockable(aux_note) else NON_SHOCKABLE

    rec_names = get_record_names(VFDB)
    rec_examples = {}  # will build this into a dict of {"rec_name": [examples]} where each example is a segment of length window_sec

    for rec_name in rec_names:
        fp = dbfp(VFDB, rec_name)

        rec = wfdb.rdrecord(fp)
        ann = wfdb.rdann(fp, "atr")
        sfreq = rec.fs

        # Apply NaN guard (VFDB has no NaNs in practice, but guard for consistency and defensibility)
        signal = rec.p_signal.copy()
        large_nan_mask = handle_signal_nans(signal, max_nan_gap=int(MAX_NAN_GAP_SEC * sfreq))

        # Get a list of intervals for each annotation. Note that the documentation does not make it clear what label
        # should be assigned to the pre-first-annotation interval, but empirically we found it's always non-shockable.
        intervals = [(NON_SHOCKABLE, (0, ann.sample[0] - 1))]
        for i in range(len(ann.sample) - 1):
            start_sample = ann.sample[i]
            end_sample = ann.sample[i + 1] - 1
            label = aux_note_to_label(ann.aux_note[i])
            intervals.append((label, (start_sample, end_sample)))

        # Add the last interval, which ends at last index
        last_label = aux_note_to_label(ann.aux_note[-1])
        intervals.append((last_label, (ann.sample[-1], len(signal) - 1)))

        # Segment the intervals
        swindow = int(window_sec * sfreq)
        examples = segment_intervals(
            full_sequence=signal,
            labeled_intervals=intervals,
            swindow=swindow,
            normalize=normalize,
            drop_mask=large_nan_mask,
        )

        # Track examples per record so later we can do a record-level split (NOT segment-level, to avoid data leakage
        # between train/valid/test sets)
        rec_examples[rec_name] = examples

    return rec_examples


def cudb__get_examples_by_record(window_sec, normalize=True):
    """
    For CUDB, check two cases:

    1)  Search for "[" to get idx1; if found, search for closing "]" to get idx2 or if missing do idx2=None. Then
        set start=sample[idx1], end=(None if idx2 is None else sample[idx2]). Then grab
        shockable_seg=rec.p_signal[start:end].

    2)  If (1) finds nothing, search for "+" in rec.symbol and grab its index (idx1). Then check aux_note[idx1] for
        a shockable label. If it's a shockable label, then find the next "+" and grab its index (idx2) or if missing
        then idx2=None. Then set start=sample[idx1], end=sample[idx2] (or None if idx2 is None). Then grab
        shockable_seg=rec.p_signal[start:end].
    """

    def is_shockable(aux_note):
        rhythm = clean_aux_note(aux_note)
        return rhythm.upper() in {'VF', 'VT'}

    rec_names = get_record_names(CUDB)
    rec_examples = {}  # will build this into a dict of {"rec_name": [examples]} where each example is a segment of length window_sec

    for rec_name in rec_names:
        fp = dbfp(CUDB, rec_name)

        rec = wfdb.rdrecord(fp)
        ann = wfdb.rdann(fp, "atr")
        sfreq = rec.fs

        signal = rec.p_signal.copy()
        large_nan_mask = handle_signal_nans(signal, max_nan_gap=int(MAX_NAN_GAP_SEC * sfreq))

        def get_interval_from_symbol(symbol, start_idx):
            end_idx = None
            for j in range(start_idx+1, len(ann.symbol)):
                if ann.symbol[j] == symbol:
                    end_idx = j
                    break
            start_samp = ann.sample[start_idx]  # sample of the opening annotation ('[' or '+')
            end_samp = len(signal) - 1 if end_idx is None else ann.sample[end_idx] # sample of the closing annotation (']' or '+')
            return end_idx, start_samp, end_samp

        # Collect shockable intervals: First search for brackets "[...]" which strictly denote VF; if record has
        # no brackets, then search for "+" which denotes rhythm change (could be VT/VF, or N)

        shockable_intervals = []
        uses_brackets = '[' in ann.symbol
        i = 0
        while i < len(ann.symbol):
            start_sample, end_sample = None, None

            # Match "[" with "]" to get a VF interval
            if ann.symbol[i] == '[':
                i, start_sample, end_sample = get_interval_from_symbol(']', i)

            # Match "+" with "+" to get a rhythm interval, could be VT/VF
            elif (not uses_brackets) and (ann.symbol[i] == '+') and is_shockable(ann.aux_note[i]):
                i, start_sample, end_sample = get_interval_from_symbol('+', i)

            # If interval was found, add it to our list of intervals
            if (start_sample is not None) and (end_sample is not None):
                shockable_intervals.append((SHOCKABLE, (start_sample, end_sample)))

            # `i` could be None if `get_interval_from_symbol` found no closing symbol, meaning the interval runs to
            # the end of the recording, so we're done and could exit the loop.
            if i is None:
                break

            # Increment symbol index
            i += 1

        # Collect non-shockable intervals (the leftover space outside shockable intervals)
        nonshockable_intervals = []
        i = 0
        for _, (start_sample, end_sample) in shockable_intervals:
            nonshockable_intervals.append((NON_SHOCKABLE, (i, start_sample)))
            i = end_sample + 1
        # Last interval
        if i < len(signal):
            nonshockable_intervals.append((NON_SHOCKABLE, (i, len(signal) - 1)))

        # Segment the intervals
        swindow = int(window_sec * sfreq)
        examples = segment_intervals(
            full_sequence=signal,
            labeled_intervals=shockable_intervals + nonshockable_intervals,
            swindow=swindow,
            normalize=normalize,
            drop_mask=large_nan_mask,
        )

        # Track examples per record so later we can do a record-level split (NOT segment-level, to avoid data leakage
        # between train/valid/test sets
        rec_examples[rec_name] = examples

    return rec_examples


def handle_signal_nans(signal, max_nan_gap):
    """
    Process NaN values in a (n_samples, n_leads) signal in-place.

    For each channel, each contiguous NaN gap is linearly interpolated or flagged
    for window-dropping. A gap is interpolated only when all three hold:

      (A) It is interior — neither leading (starting at sample 0) nor trailing
          (ending at the last sample), so valid signal exists on both sides.
      (B) Its length is <= max_nan_gap.
      (C) The 2*gap_len samples immediately following the gap are NaN-free, or
          the signal ends before that check window is exhausted (accepted near
          end-of-record — handles the case where fewer than 2*gap_len samples
          remain but the gap is still genuinely isolated).

    This guards against interpolating across clustered small gaps where tiny
    islands of valid signal are surrounded by NaN — in that case the gap is
    small by condition B but condition C fails because the next NaN gap is
    too close, indicating the overall region is majority-NaN and meaningless
    to interpolate.

    Leading/trailing NaN regions are left as NaN and dropped (never
    extrapolated), since there is no valid signal on one side to anchor from.

    All non-interpolated gaps are OR-ed into large_nan_mask so that any
    segmentation window overlapping them is dropped.

    Returns large_nan_mask: bool array of shape (n_samples,).
    Modifies signal in-place for interpolated gaps.
    """
    n_samples = signal.shape[0]
    large_nan_mask = np.zeros(n_samples, dtype=bool)

    for ch in range(signal.shape[1]):
        col = signal[:, ch]          # view; writes to col update signal directly
        nan_mask = np.isnan(col)     # original NaN pattern — used for gap detection and condition C
        if not nan_mask.any():
            continue

        # Detect all contiguous NaN gaps
        changes = np.diff(nan_mask.astype(int))
        gap_starts = np.where(changes == 1)[0] + 1
        gap_ends = np.where(changes == -1)[0] + 1
        if nan_mask[0]:
            gap_starts = np.concatenate([[0], gap_starts])
        if nan_mask[-1]:
            gap_ends = np.concatenate([gap_ends, [n_samples]])

        for s, e in zip(gap_starts, gap_ends):
            gap_len = e - s
            is_leading = (s == 0)
            is_trailing = (e == n_samples)
            is_small = (gap_len <= max_nan_gap)
            # Condition C: require 2 x gap_len clean samples after the gap, unless
            # the signal ends before the check window is exhausted (accepted near
            # end-of-record). Uses original nan_mask, not the progressively-filled col.
            strip_ends_after = (e + 2 * gap_len > n_samples)
            clean_after = strip_ends_after or (not nan_mask[e: e + 2 * gap_len].any())

            if not is_leading and not is_trailing and is_small and clean_after:
                # Interior, small, well-isolated gap: linearly interpolate.
                # col is a view updated in-place, so previously filled gaps serve
                # as valid anchors for interpolating subsequent gaps.
                indices = np.arange(n_samples)
                valid = ~np.isnan(col)
                col[s:e] = np.interp(np.arange(s, e), indices[valid], col[valid])
            else:
                # Leading/trailing, too large, or clustered gap: flag for dropping.
                large_nan_mask[s:e] = True

    return large_nan_mask


def segment_intervals(
        full_sequence,
        labeled_intervals,
        swindow,
        lead=0,
        normalize=True,
        ignore_partial=True,
        drop_mask=None
):
    """
    Slice a (n_samples, n_leads) or (n_samples,) signal into fixed-length windows.
    `lead` selects which column to extract when the signal is multi-lead.
    VFDB has 2 leads (both labeled 'ECG'); we use lead 0 by convention.
    CUDB has 1 lead; lead 0 is the only option.
    `drop_mask` is an optional boolean array of shape (n_samples,); any window that
    overlaps a True sample is dropped (used to exclude uninterpolatable NaN gaps).
    """
    # Extract single lead if signal is 2D
    if full_sequence.ndim == 2:
        full_sequence = full_sequence[:, lead]

    examples = []
    for label, (start, end) in labeled_intervals:
        # We intentionally avoid overlap so as not to distort evaluation metrics (duplicate evidence)
        segments = [full_sequence[i: i + swindow] for i in range(start, end + 1, swindow)]

        # Remove partial segments if ignore_partial==True
        if ignore_partial and len(segments) > 0 and len(segments[-1]) != swindow:
            segments.pop()

        # Drop windows overlapping large NaN gaps
        if drop_mask is not None:
            starts = range(start, end + 1, swindow)[:len(segments)]
            segments = [seg for seg, s in zip(segments, starts) if not drop_mask[s: s + swindow].any()]

        if normalize:
            segments = list(map(zscore, segments))

        examples.extend([(seg, label) for seg in segments])

    return examples


def zscore(seg):
    return (seg - seg.mean()) / max(seg.std(), 1e-6)


# DATASET SPLITTING ----------------------------------------------------------------------------------------------------


def partition_records_balanced(rec_examples, target_proportion):
    """
    Finds the record subset that minimizes combined error over both classes:

        |shock_sum/total_shock - p| + |noshock_sum/total_noshock - p|

    Brute-forces all 2^n subsets with numpy vectorization (chunked to keep
    memory ~25 MB per chunk). For VFDB (n=22) this is ~4M subsets and runs
    fairly quickly.
    """
    if not 0 <= target_proportion <= 1:
        raise ValueError("target_proportion must be between 0 and 1")

    def num_shockable_examples(examples):
        return sum(1 for _, label in examples if label == SHOCKABLE)

    def num_nonshockable_examples(examples):
        return sum(1 for _, label in examples if label == NON_SHOCKABLE)

    records = list(rec_examples.items())
    n = len(records)

    shock_w = np.array([num_shockable_examples(exs) for _, exs in records], dtype=np.float32)
    noshock_w = np.array([num_nonshockable_examples(exs) for _, exs in records], dtype=np.float32)
    total_shock = shock_w.sum()
    total_noshock = noshock_w.sum()

    bit_positions = np.arange(n, dtype=np.int32)
    chunk_size = 1 << 18   # 262,144 subsets per chunk -> ~25 MB of working memory
    n_subsets = 1 << n

    best_error = np.inf
    best_mask = 0

    for chunk_start in range(0, n_subsets, chunk_size):
        masks = np.arange(chunk_start, min(chunk_start + chunk_size, n_subsets), dtype=np.int32)

        # bits[i, j] = 1 if record j is included in subset i
        bits = ((masks[:, None] >> bit_positions) & 1).astype(np.float32)  # (chunk, n)

        shock_sums = bits @ shock_w  # (chunk,)
        noshock_sums = bits @ noshock_w  # (chunk,)

        errors = (
                np.abs(shock_sums / total_shock - target_proportion)
                + np.abs(noshock_sums / total_noshock - target_proportion)
        )

        idx = int(np.argmin(errors))
        if errors[idx] < best_error:
            best_error = float(errors[idx])
            best_mask  = int(masks[idx])

    # Reconstruct which records are in the best subset.
    # Both dicts are built by iterating rec_examples (stable insertion order) so that
    # valid_set / train_set assembly is deterministic. subset_names is kept as a set
    # for O(1) membership tests only.
    subset_names = {records[i][0] for i in range(n) if (best_mask >> i) & 1}

    target_examples = {name: rec_examples[name] for name in rec_examples if name in subset_names}
    complement_examples = {name: rec_examples[name] for name in rec_examples if name not in subset_names}

    target_shock = sum(num_shockable_examples(exs) for exs in target_examples.values())
    target_noshock = sum(num_nonshockable_examples(exs) for exs in target_examples.values())
    target_count = sum(len(exs) for exs in target_examples.values())
    total_count = sum(len(exs) for exs in rec_examples.values())

    return {
        'target_examples': target_examples,
        'complement_examples': complement_examples,
        'shockable_proportion': target_shock / total_shock if total_shock > 0 else 0.0,
        'nonshockable_proportion': target_noshock / total_noshock if total_noshock > 0 else 0.0,
        'overall_proportion': target_count / total_count if total_count > 0 else 0.0
    }


def train_valid_test_split(
        vfdb_examples,
        cudb_examples,
        vfdb_valid_p,
        shuffle=True,
        verbose=False
):
    vprint = print if verbose else (lambda *_, **__: None)

    # We intentionally split by record (not by segment) to avoid leaking intra-record correlations from training
    # to test data, or even from training to validation (not as bad, but still worth accounting for)

    # Use all of CUDB for the test set
    test_set = [ex for examples in cudb_examples.values() for ex in examples]

    # Split VFDB (80% train, 20% valid)
    split_info = partition_records_balanced(vfdb_examples, target_proportion=vfdb_valid_p)
    valid_set = [ex for examples in split_info['target_examples'].values() for ex in examples]
    train_set = [ex for examples in split_info['complement_examples'].values() for ex in examples]

    # Check split proportions
    vprint(f'VFDB Validation Set Proportion (desired proportion = {vfdb_valid_p}):')
    vprint(f'  - shockable:      {100 * split_info['shockable_proportion'] :.2f}%')
    vprint(f'  - non-shockable:  {100 * split_info['nonshockable_proportion'] :.2f}%')
    vprint(f'  - overall:        {100 * split_info['overall_proportion'] :.2f}%')
    vprint()

    train_pos = sum(1 for _, label in train_set if label == SHOCKABLE)
    train_neg = sum(1 for _, label in train_set if label == NON_SHOCKABLE)
    valid_pos = sum(1 for _, label in valid_set if label == SHOCKABLE)
    valid_neg = sum(1 for _, label in valid_set if label == NON_SHOCKABLE)
    test_pos = sum(1 for _, label in test_set if label == SHOCKABLE)
    test_neg = sum(1 for _, label in test_set if label == NON_SHOCKABLE)

    n_train_rec = len(split_info['complement_examples'])
    n_valid_rec = len(split_info['target_examples'])
    n_test_rec = len(cudb_examples)

    rows = [
        ('Split',   'Database', 'Records',      'Shockable',    'Non-Shockable',    'Ratio'),
        ('Train',   'VFDB',     n_train_rec,    train_pos,      train_neg,          f"1:{train_neg/train_pos:.1f}"),
        ('Valid',   'VFDB',     n_valid_rec,    valid_pos,      valid_neg,          f"1:{valid_neg/valid_pos:.1f}"),
        ('Test',    'CUDB',     n_test_rec,     test_pos,       test_neg,           f"1:{test_neg/test_pos:.1f}")
    ]

    fmt_cell = lambda v: f"{v:,}" if isinstance(v, int) else str(v)
    col_w = [max(len(fmt_cell(r[c])) for r in rows) for c in range(len(rows[0]))]
    sep = '  '
    div = '-' * (sum(col_w) + len(sep) * (len(col_w) - 1))
    vprint(div)
    for i, row in enumerate(rows):
        line = sep.join(fmt_cell(row[c]).ljust(col_w[c]) if c < 2 else fmt_cell(row[c]).rjust(col_w[c]) for c in range(len(row)))
        vprint(line)
        if i == 0:
            vprint(div)
    vprint(div)

    if shuffle:
        # Create RNG here to ensure we always get the same shuffle order
        _shuffle_rng = random.Random(RANDOM_SEED)
        _shuffle_rng.shuffle(train_set)
        _shuffle_rng.shuffle(valid_set)
        _shuffle_rng.shuffle(test_set)

    return {
        'train': train_set,
        'valid': valid_set,
        'test': test_set
    }


# SAVE / LOAD DATA -----------------------------------------------------------------------------------------------------


class ECGDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, L) float32 -> unsqueeze to (N, 1, L) for 1D CNN (batch, channels, length)
        # y: (N,) int64
        # NOTE: `L = window_sec * sfreq` is the segment length used in the data preprocessing stage
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def save_data_splits(splits_dict, *, path=SPLITS_NPZ_PATH):
    np.savez_compressed(
        path,
        train_X=np.array([seg for seg, _ in splits_dict['train']], dtype=np.float32),
        train_y=np.array([label for _, label in splits_dict['train']], dtype=np.int64),
        valid_X=np.array([seg for seg, _ in splits_dict['valid']], dtype=np.float32),
        valid_y=np.array([label for _, label in splits_dict['valid']], dtype=np.int64),
        test_X=np.array([seg for seg, _ in splits_dict['test']], dtype=np.float32),
        test_y=np.array([label for _, label in splits_dict['test']], dtype=np.int64),
    )


def load_data_splits(*, path=SPLITS_NPZ_PATH, batch_size=BATCH_SIZE, rng=None):
    data = np.load(path)
    # Shuffle RNG is passed in as needed (intentional design choice; at the start of training we can re-seed
    # the RNG each epoch so the shuffle is deterministic per epoch and thus interruption-safe).
    train_loader = DataLoader(ECGDataset(data['train_X'], data['train_y']), batch_size=batch_size, shuffle=True, generator=rng)
    valid_loader = DataLoader(ECGDataset(data['valid_X'], data['valid_y']), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(ECGDataset(data['test_X'],  data['test_y']),  batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


# DRIVER ---------------------------------------------------------------------------------------------------------------


def main():
    print('Preprocessing data...\n')

    # Preprocess data to get labeled/segmented/normalized examples from VFDB and CUDB
    vfdb_examples = vfdb__get_examples_by_record(window_sec=WINDOW_SEC, normalize=NORMALIZE)
    cudb_examples = cudb__get_examples_by_record(window_sec=WINDOW_SEC, normalize=NORMALIZE)

    # Split examples into train, valid, test splits
    splits_dict = train_valid_test_split(
        vfdb_examples=vfdb_examples,
        cudb_examples=cudb_examples,
        vfdb_valid_p=VFDB_VALID_P,
        shuffle=True,
        verbose=True
    )

    # Save the data splits to be loaded later
    save_data_splits(splits_dict)


if __name__ == '__main__':
    main()
