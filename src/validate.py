"""Check the training data before anything is trained on it, in two stages.

- validate_raw_data: the CSV as read, before cleaning. Catches what cleaning would hide or crash on
  (missing columns, text in a magnitude column, too many placeholder rows).
- validate_training_data: the cleaned DataFrame the model will train on.

Each collects every problem and raises one ValueError listing them all, so a broken data version
stops training instead of becoming a model.
"""
import numpy as np
import pandas as pd

from src.config import (
    CLASSES,
    FILTERS,
    MAG_MAX,
    MAG_MIN,
    MAX_PLACEHOLDER_SHARE,
    MIN_ROWS,
    PLACEHOLDER,
    TARGET,
)


def _raise_if_problems(problems, stage):
    if problems:
        raise ValueError(f'{stage} failed validation:\n  - ' + '\n  - '.join(problems))


def validate_raw_data(df):
    """Raise ValueError if the CSV as read can't be cleaned safely."""
    problems = []

    # 1. Required columns (cleaning reads the magnitude columns, so check before it runs)
    missing = [c for c in FILTERS + [TARGET] if c not in df.columns]
    if missing:
        problems.append(f'missing columns: {missing}')

    # 2. Magnitudes are numbers. Text like 'abc' turns the whole column into text, and the numeric
    #    checks would crash on it. ('n/a' and empty cells are read as missing and caught later.)
    numeric = []
    for col in [c for c in FILTERS if c in df.columns]:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
            continue
        as_numbers = pd.to_numeric(df[col], errors='coerce')
        not_numbers = df[col][as_numbers.isna() & df[col].notna()]
        problems.append(f'{len(not_numbers)} rows have non-numeric {col} (e.g. {not_numbers.iloc[0]!r})')

    # 3. Few placeholder rows. Cleaning drops them, so this is the only place a mass loss shows up.
    if numeric and len(df):
        placeholder_rows = (df[numeric] == PLACEHOLDER).any(axis=1).sum()
        if placeholder_rows / len(df) > MAX_PLACEHOLDER_SHARE:
            problems.append(f'{placeholder_rows} of {len(df)} rows ({placeholder_rows / len(df):.1%}) '
                            f'have the {PLACEHOLDER} placeholder, more than {MAX_PLACEHOLDER_SHARE:.0%}')

    _raise_if_problems(problems, 'Raw training data')


def validate_training_data(df):
    """Raise ValueError if the cleaned data is unfit for training; print one line if it passes."""
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

    _raise_if_problems(problems, 'Training data')

    print(f'Validation passed: {len(df)} rows, {df[TARGET].nunique()} classes')
