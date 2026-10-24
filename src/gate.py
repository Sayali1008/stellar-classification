"""Deploy gate: decide whether a newly trained model may replace the live one.

Champion/challenger on the same test set: the live model (the champion) is downloaded from S3 and
scored on the new model's test split, so both numbers come from the same objects even when the data
version changes. The new model (the challenger) passes if it reaches GATE_MIN_F1 and scores at least
the live model's macro F1 minus F1_TOLERANCE.

If the live model can't be compared (site down, not in S3, saved with another scikit-learn), only
the floor applies, and the reason is printed. Used by `train.py --deploy` (deploy/deploy.sh).
"""
import json
import tempfile
import urllib.request
from pathlib import Path

from sklearn.metrics import f1_score

from src.config import (
    DATA_BUCKET,
    GATE_MIN_F1,
    F1_TOLERANCE,
    LIVE_URL,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_PREFIX,
)
from src.predict import read_model


def decide(new_f1, live_f1):
    """(passed, reason) for a new model's macro F1 against the floor and, if known, the live model's."""
    if new_f1 < GATE_MIN_F1:
        return False, f'new macro F1 {new_f1:.4f} is below the floor {GATE_MIN_F1}'
    if live_f1 is None:
        return True, f'new macro F1 {new_f1:.4f} reaches the floor {GATE_MIN_F1} (no live model to compare)'
    if new_f1 < live_f1 - F1_TOLERANCE:
        return False, (f'new macro F1 {new_f1:.4f} is worse than the live model\'s {live_f1:.4f} '
                       f'by more than {F1_TOLERANCE}')
    return True, f'new macro F1 {new_f1:.4f} vs live {live_f1:.4f} (tolerance {F1_TOLERANCE})'


def live_model_version():
    """The model version the live API reports at /health."""
    # A long timeout: the first request after a quiet period waits for a Lambda cold start
    with urllib.request.urlopen(f'{LIVE_URL}/health', timeout=30) as response:
        return json.load(response)['model_version']


def download_live_model(version):
    """The live model and its metadata, downloaded from s3://DATA_BUCKET/MODELS_PREFIX/<version>/."""
    # Imported here, as in src/data.py: the test install has no boto3
    import boto3

    s3 = boto3.client('s3')
    folder = Path(tempfile.mkdtemp())
    for name in ('model.joblib', 'metadata.json'):
        s3.download_file(DATA_BUCKET, f'{MODELS_PREFIX}/{version}/{name}', str(folder / name))
    return read_model(folder / 'model.joblib', folder / 'metadata.json')


def run_gate(metadata, X_test, y_test):
    """Score the live model on this test split and decide. Returns a dict for metrics.json."""
    new_f1 = metadata['test_metrics']['macro_f1']
    live_version, live_f1, note = None, None, None
    try:
        live_version = live_model_version()
        live_model, _ = download_live_model(live_version)
        live_f1 = float(f1_score(y_test, live_model.predict(X_test), average='macro'))
    except Exception as error:
        # Any failure to compare leaves the floor as the only check; say why in the log
        note = f'live model not compared ({type(error).__name__}: {error}); floor only'
        print(f'Gate: {note}')

    passed, reason = decide(new_f1, live_f1)
    print(f"Gate {'PASSED' if passed else 'FAILED'}: {reason}")
    return {
        'model_version': metadata['model_version'],
        'data_version': metadata['data_version'],
        'new_macro_f1': new_f1,
        'live_model_version': live_version,
        'live_macro_f1_same_test_set': live_f1,
        'min_f1': GATE_MIN_F1,
        'tolerance': F1_TOLERANCE,
        'passed': passed,
        'reason': reason,
        'note': note,
    }


def upload_model(gate_result):
    """Store the passing model in s3://DATA_BUCKET/MODELS_PREFIX/<version>/ with its gate result."""
    import boto3

    s3 = boto3.client('s3')
    prefix = f"{MODELS_PREFIX}/{gate_result['model_version']}"
    s3.upload_file(str(MODEL_PATH), DATA_BUCKET, f'{prefix}/model.joblib')
    s3.upload_file(str(METADATA_PATH), DATA_BUCKET, f'{prefix}/metadata.json')
    s3.put_object(Bucket=DATA_BUCKET, Key=f'{prefix}/metrics.json',
                  Body=json.dumps(gate_result, indent=2).encode())
    print(f'Model stored in s3://{DATA_BUCKET}/{prefix}/')

