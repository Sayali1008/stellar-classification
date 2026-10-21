"""Training-data validation: each kind of broken data stops training with a message that names it.

Every test breaks a copy of the shared synthetic data in one way and checks the message. Most call
the validators directly; the load_data tests at the end go through the CSV, as training does.
"""
import io
import re

import numpy as np
import pytest

from src.config import MAG_MAX, MAG_MIN, MAX_PLACEHOLDER_SHARE, MIN_ROWS, PLACEHOLDER, TARGET
from src.data import load_data
from src.validate import validate_raw_data, validate_training_data


def test_good_data_passes_both_stages(synthetic_df):
    validate_raw_data(synthetic_df)
    validate_training_data(synthetic_df)


# ---- Stage 1: the CSV as read, before cleaning ----

def test_raw_rejects_an_empty_file(synthetic_df):
    with pytest.raises(ValueError, match='file has no data rows'):
        validate_raw_data(synthetic_df.head(0))


def test_raw_rejects_a_missing_column(synthetic_df):
    with pytest.raises(ValueError, match=r"missing columns: \['g'\]"):
        validate_raw_data(synthetic_df.drop(columns='g'))


def test_raw_rejects_text_in_a_magnitude_column(synthetic_df):
    # One bad cell makes pandas read the whole column as text, as it would from the CSV
    df = synthetic_df.astype({'r': object})
    df.loc[[3, 4, 5], 'r'] = 'abc'
    with pytest.raises(ValueError, match=r"3 rows have non-numeric r \(e.g. 'abc'\)"):
        validate_raw_data(df)


def placeholder_rows(df, count):
    """Copy of df with the -9999 placeholder in `u` on the first `count` rows."""
    df = df.copy()
    df.loc[:count - 1, 'u'] = PLACEHOLDER
    return df


def test_raw_allows_placeholders_up_to_the_limit(synthetic_df):
    # Exactly at the limit is still allowed: cleaning drops these rows
    at_limit = int(len(synthetic_df) * MAX_PLACEHOLDER_SHARE)
    validate_raw_data(placeholder_rows(synthetic_df, at_limit))


def test_raw_rejects_placeholders_over_the_limit(synthetic_df):
    over_limit = int(len(synthetic_df) * MAX_PLACEHOLDER_SHARE) + 1
    with pytest.raises(ValueError, match=f'have the {PLACEHOLDER} placeholder'):
        validate_raw_data(placeholder_rows(synthetic_df, over_limit))


def test_raw_reports_every_problem_at_once(synthetic_df):
    df = synthetic_df.drop(columns='g').astype({'r': object})
    df.loc[0, 'r'] = 'abc'
    with pytest.raises(ValueError) as error:
        validate_raw_data(df)
    assert 'missing columns' in str(error.value)
    assert 'non-numeric r' in str(error.value)


# ---- Stage 2: the cleaned data the model trains on ----

def test_training_rejects_too_few_rows(synthetic_df):
    with pytest.raises(ValueError, match=f'expected at least {MIN_ROWS}'):
        validate_training_data(synthetic_df.head(MIN_ROWS - 1))


def test_training_rejects_an_unknown_class(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, TARGET] = 'galaxy'
    with pytest.raises(ValueError, match=r"unknown classes: \['galaxy'\]"):
        validate_training_data(df)


def test_training_rejects_a_missing_class(synthetic_df):
    df = synthetic_df[synthetic_df[TARGET] != 'QSO']
    with pytest.raises(ValueError, match=r"missing classes: \['QSO'\]"):
        validate_training_data(df)


def test_training_rejects_missing_values(synthetic_df):
    df = synthetic_df.copy()
    df.loc[[0, 1], 'z'] = np.nan
    with pytest.raises(ValueError, match='2 rows have a missing or infinite z'):
        validate_training_data(df)


def test_training_rejects_out_of_range_magnitudes(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, 'i'] = 99
    with pytest.raises(ValueError, match=re.escape(f'1 rows have i outside [{MAG_MIN}, {MAG_MAX}]')):
        validate_training_data(df)


# ---- load_data: read the CSV, validate it, drop placeholder rows ----

def as_csv(df):
    """df written out as CSV text, as an open stream: what load_data receives from S3."""
    return io.StringIO(df.to_csv(index=False))


def test_load_data_drops_placeholder_rows(synthetic_df):
    df = load_data(as_csv(placeholder_rows(synthetic_df, 5)))
    assert len(df) == len(synthetic_df) - 5
    assert not (df['u'] == PLACEHOLDER).any()


def test_load_data_rejects_text_read_from_the_csv(synthetic_df):
    # Here pandas itself turns the column into text while reading, as with a real broken file
    df = synthetic_df.astype({'r': object})
    df.loc[0, 'r'] = 'abc'
    with pytest.raises(ValueError, match='non-numeric r'):
        load_data(as_csv(df))
