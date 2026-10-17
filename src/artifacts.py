"""Save the trained pipeline and the metadata that travels with it.

No plotting imports here: the API loads artifacts too, and the serving image has no matplotlib.
"""
import json
from datetime import UTC, datetime

import joblib
import sklearn
from sklearn.metrics import classification_report

from src.config import METADATA_PATH, MODEL_PATH


def build_metadata(model, y_test, test_preds, data_version):
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    return {
        # UTC training time: unique, sorts in time order, needs no stored counter
        'model_version': f'{datetime.now(UTC):%Y%m%d-%H%M%S}',
        # A saved model is only guaranteed to load in the same scikit-learn version
        'sklearn_version': sklearn.__version__,
        # Scored on the held-out test set; the API never sees that data, so record it now
        'test_metrics': {
            'macro_f1': float(report['macro avg']['f1-score']),
            'per_class': {
                str(c): {k: float(report[c][k]) for k in ('precision', 'recall', 'f1-score', 'support')}
                for c in model.classes_
            },
        },
        'data_version': data_version,
    }


def save_artifacts(model, metadata):
    """Write model.joblib (compressed) and metadata.json. Returns the model file size in MB."""
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return MODEL_PATH.stat().st_size / 1e6
