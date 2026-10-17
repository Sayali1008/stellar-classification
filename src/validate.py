"""Check the whole training DataFrame before anything is trained on it.

Runs on the cleaned data (after load_data). Collects every problem and raises one ValueError
listing them all, so a broken data version stops training instead of becoming a model.
"""
import numpy as np

from src.config import CLASSES, FILTERS, MAG_MAX, MAG_MIN, MIN_ROWS, TARGET


def validate_training_data(df):
    """Raise ValueError if the data is unfit for training; print one line if it passes."""
    problems = []

    # 1. Required columns
    missing = [c for c in FILTERS + [TARGET] if c not in df.columns]
    if missing:
        problems.append(f'missing columns: {missing}')

    # 2. Enough rows (catches a truncated upload)
    if len(df) < MIN_ROWS:
        problems.append(f'only {len(df)} rows, expected at least {MIN_ROWS}')

    # 3. Exactly the known classes: no unknown labels, and every class present
    if TARGET in df.columns:
        found = set(df[TARGET].unique())
        unknown = found - set(CLASSES)
        absent = set(CLASSES) - found
        if unknown:
            problems.append(f'unknown classes: {sorted(unknown, key=str)}')
        if absent:
            problems.append(f'missing classes: {sorted(absent)}')

    # 4 and 5. Every magnitude is a real number, and inside a plausible range
    for col in [c for c in FILTERS if c in df.columns]:
        values = df[col]
        finite = np.isfinite(values)
        if (~finite).any():
            problems.append(f'{(~finite).sum()} rows have a missing or infinite {col}')
        out_of_range = finite & ~values.between(MAG_MIN, MAG_MAX)
        if out_of_range.any():
            problems.append(f'{out_of_range.sum()} rows have {col} outside [{MAG_MIN}, {MAG_MAX}]')

    if problems:
        raise ValueError('Training data failed validation:\n  - ' + '\n  - '.join(problems))

    print(f'Validation passed: {len(df)} rows, {df[TARGET].nunique()} classes')
