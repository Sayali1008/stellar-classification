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


@lru_cache(maxsize=1)
def load_model():
    """Read model.joblib and metadata.json from disk on the first call; later calls reuse them."""
    metadata = json.loads(METADATA_PATH.read_text())
    # A pickled scikit-learn model is only guaranteed to load correctly in the version that saved it
    if metadata['sklearn_version'] != sklearn.__version__:
        raise RuntimeError(
            f"Model was saved with scikit-learn {metadata['sklearn_version']}, "
            f"but {sklearn.__version__} is installed. Retrain or pin the same version."
        )
    model = joblib.load(MODEL_PATH)
    return model, metadata


def model_version():
    return load_model()[1]['model_version']


def predict(objects: list[StellarObject]) -> list[Prediction]:
    """Classify a batch of objects in one call. Returns one Prediction per object, in the same order."""
    if not objects:
        return []
    model, _ = load_model()

    # The pipeline selects columns by name, so it needs a DataFrame, not a NumPy array
    X = pd.DataFrame([obj.model_dump() for obj in objects], columns=FILTERS)
    probas = model.predict_proba(X)

    # predict_proba columns follow model.classes_ (alphabetical), not config.CLASSES
    classes = [str(c) for c in model.classes_]
    return [
        Prediction(
            predicted_class=classes[row.argmax()],
            probabilities={c: round(float(p), 3) for c, p in zip(classes, row)},
        )
        for row in probas
    ]
