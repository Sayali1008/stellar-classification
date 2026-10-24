"""Load the saved model once and turn validated objects into predictions.

Serving code: imports only config and schemas, never the training modules (src.data, src.model,
src.evaluate), because the Lambda image has no matplotlib or training data.
"""
import json
from functools import lru_cache

import joblib
import pandas as pd
import sklearn

from src.config import FILTERS, METADATA_PATH, MODEL_PATH
from src.schemas import Prediction, StellarObject


def read_model(model_path, metadata_path):
    """Read a saved model and its metadata, refusing a model saved with another scikit-learn.

    Used by load_model below (the API's model) and by the deploy gate (the live model from S3).
    """
    metadata = json.loads(metadata_path.read_text())
    # A pickled scikit-learn model is only guaranteed to load correctly in the version that saved it
    if metadata['sklearn_version'] != sklearn.__version__:
        raise RuntimeError(
            f"Model was saved with scikit-learn {metadata['sklearn_version']}, "
            f"but {sklearn.__version__} is installed. Retrain or pin the same version."
        )
    model = joblib.load(model_path)
    return model, metadata


@lru_cache(maxsize=1)
def load_model():
    """Read model.joblib and metadata.json from disk on the first call; later calls reuse them."""
    return read_model(MODEL_PATH, METADATA_PATH)


def model_version():
    return load_model()[1]['model_version']


def range_warnings(obj, stats):
    """One message per filter of `obj` outside the training range in `stats` (metadata's input_stats).

    Models saved before input_stats existed (e.g. after a rollback) have no stats: no warnings.
    Magnitudes work backwards, so a value below the range is brighter, above it fainter.
    """
    if not stats:
        return []
    low_q, high_q = stats['range_quantiles']
    warnings = []
    for f, (low, high) in stats['ranges'].items():
        value = obj[f]
        if value < low:
            warnings.append(f'{f} = {value:.2f} is brighter than {1 - low_q:.0%} of training objects '
                            f'(training range {low:.2f}–{high:.2f})')
        elif value > high:
            warnings.append(f'{f} = {value:.2f} is fainter than {high_q:.0%} of training objects '
                            f'(training range {low:.2f}–{high:.2f})')
    return warnings


def predict(objects: list[StellarObject]) -> list[Prediction]:
    """Classify a batch of objects in one call. Returns one Prediction per object, in the same order."""
    if not objects:
        return []
    model, metadata = load_model()
    stats = metadata.get('input_stats')

    # The pipeline selects columns by name, so it needs a DataFrame, not a NumPy array
    rows = [obj.model_dump() for obj in objects]
    X = pd.DataFrame(rows, columns=FILTERS)
    probas = model.predict_proba(X)

    # predict_proba columns follow model.classes_ (alphabetical), not config.CLASSES
    classes = [str(c) for c in model.classes_]
    return [
        Prediction(
            predicted_class=classes[proba.argmax()],
            probabilities={c: round(float(p), 3) for c, p in zip(classes, proba)},
            warnings=range_warnings(row, stats),
        )
        for row, proba in zip(rows, probas)
    ]
