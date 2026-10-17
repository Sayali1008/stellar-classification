import boto3
import pandas as pd

from src.config import DATA_BUCKET, DATA_KEY, DATA_PATH, FILTERS, ID_COLS, PLACEHOLDER


def load_data(path=DATA_PATH):
    """Load the SDSS17 CSV and apply the cleaning steps from the analysis notebook."""
    # converting ID columns to string
    df = pd.read_csv(path, dtype={c: str for c in ID_COLS})

    # convert MJD to standard datetime
    df['Date'] = pd.to_datetime(df['MJD'], unit='D', origin='1858-11-17')

    # Drop rows with the -9999 placeholder in any filter (1 row in this dataset).
    # Placeholder, not a real measurement; 0.001% of data.
    df = df[(df[FILTERS] != PLACEHOLDER).all(axis=1)]

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
