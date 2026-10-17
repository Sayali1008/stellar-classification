import boto3
import pandas as pd

from src.config import DATA_BUCKET, DATA_KEY, FILTERS, ID_COLS, PLACEHOLDER
from src.validate import validate_raw_data


def load_data(source):
    """Read the SDSS17 CSV (a file path or an open stream), check it, and apply the cleaning steps."""
    # converting ID columns to string
    df = pd.read_csv(source, dtype={c: str for c in ID_COLS})

    # Check before cleaning: cleaning reads the magnitude columns and drops placeholder rows,
    # so a broken file would otherwise crash here or shrink without a warning
    validate_raw_data(df)

    # Drop rows with the -9999 placeholder in any filter (1 row in this dataset).
    # Placeholder, not a real measurement; 0.001% of data.
    rows_before = len(df)
    df = df[(df[FILTERS] != PLACEHOLDER).all(axis=1)]
    print(f'Dropped {rows_before - len(df)} rows with the {PLACEHOLDER} placeholder')

    return df


def read_training_data():
    """Download the training CSV from S3, clean it, and return (data_version, df)."""
    s3 = boto3.client('s3')
    response = s3.get_object(
        Bucket=DATA_BUCKET,
        Key=DATA_KEY,
    )

    print(f'Data: s3://{DATA_BUCKET}/{DATA_KEY} (version {response["VersionId"]})')
    df = load_data(response['Body'])

    return response['VersionId'], df
