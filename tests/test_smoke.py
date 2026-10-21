"""Smoke test on the real data: does the training pipeline still produce a good model?

The other tests check behaviour on synthetic data; this one checks quality. It needs the private
training data in S3, so it is skipped when boto3 isn't installed (the lean CI install) or no AWS
credentials work (e.g. not signed in with `aws login --profile stellar`).
"""
import pytest
from sklearn.metrics import f1_score

from src.config import DATA_BUCKET, DATA_KEY, FILTERS

# Lowest acceptable test macro F1. The deployed model scores 0.851; anything under 0.80 means
# something broke (data, features, settings), not ordinary noise.
MIN_MACRO_F1 = 0.80


@pytest.fixture(scope='module')
def real_data():
    """(data_version, df) from S3, or skip the test with the reason S3 can't be read."""
    boto3 = pytest.importorskip('boto3', reason='boto3 not installed (lean test install)')
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        # A cheap check before downloading: can these credentials see the training file?
        boto3.client('s3').head_object(Bucket=DATA_BUCKET, Key=DATA_KEY)
    except (BotoCoreError, ClientError) as error:
        pytest.skip(f'cannot read s3://{DATA_BUCKET}/{DATA_KEY} '
                    f'(sign in and set AWS_PROFILE=stellar): {type(error).__name__}')

    from src.data import read_training_data
    return read_training_data()


def test_real_model_meets_the_quality_bar(real_data):
    # The same steps as train.py (validate, split, build, fit, score), without saving or MLflow
    from src.model import build_model, split_data
    from src.validate import validate_training_data

    _, df = real_data
    validate_training_data(df)
    X_train, X_test, y_train, y_test = split_data(df, FILTERS)

    model = build_model().fit(X_train, y_train)
    macro_f1 = f1_score(y_test, model.predict(X_test), average='macro')

    assert macro_f1 > MIN_MACRO_F1, f'test macro F1 {macro_f1:.3f} is below {MIN_MACRO_F1}'
