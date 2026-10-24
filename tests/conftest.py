"""Shared test fixtures: synthetic data, a tiny model trained on it, and that model saved to a temp folder.

The tests never use the real data (private, in S3) or the real artifacts/ (not in git), so they run
the same on any machine, offline. They check behaviour (shapes, order, status codes), not model
quality; tests/test_smoke.py checks quality on the real data.
"""
import numpy as np
import pandas as pd
import pytest

import src.artifacts
import src.predict
from src.artifacts import build_metadata, save_artifacts
from src.config import FILTERS, TARGET
from src.model import build_model

# Per-class medians from REPORT section 4: (u_g, g_r, r_i, i_z, r)
CLASS_CENTERS = {
    'GALAXY': (1.64, 1.40, 0.72, 0.39, 20.10),
    'STAR': (1.32, 0.53, 0.21, 0.12, 18.96),
    'QSO': (0.39, 0.21, 0.15, 0.16, 20.77),
}
# Half the real spread, so the classes overlap less and the tiny model's answers are stable
SPREAD = (0.6, 0.3, 0.2, 0.2, 1.0)

# Above MIN_ROWS (10,000), so the synthetic data passes validate_training_data
ROWS_PER_CLASS = 4000

# Small forest: trains in about a second and still predicts sensibly on the synthetic classes
TINY_FOREST = {'n_estimators': 10, 'min_samples_leaf': 5}


def make_synthetic_data(rows_per_class=ROWS_PER_CLASS, seed=0):
    """Fake SDSS rows: u, g, r, i, z and class, built from colors so each class has its real shape.

    Draws each class's colors and r around its median, then works back to the five magnitudes
    (g = r + g_r, u = g + u_g, i = r - r_i, z = i - i_z). Same seed, same data.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for cls, center in CLASS_CENTERS.items():
        u_g, g_r, r_i, i_z, r = rng.normal(center, SPREAD, size=(rows_per_class, 5)).T
        g = r + g_r
        i = r - r_i
        parts.append(pd.DataFrame({
            'u': g + u_g, 'g': g, 'r': r, 'i': i, 'z': i - i_z, TARGET: cls,
        }))
    # Shuffle so the classes are mixed, as in the real file
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def center_object(cls):
    """The five magnitudes at a class's median colors and r: the most typical object of that class."""
    u_g, g_r, r_i, i_z, r = CLASS_CENTERS[cls]
    g, i = r + g_r, r - r_i
    return {'u': g + u_g, 'g': g, 'r': r, 'i': i, 'z': i - i_z}


@pytest.fixture(scope='session')
def class_centers():
    """{'GALAXY': {u, g, r, i, z}, ...}: one typical object per class, which tiny_model gets right."""
    return {cls: center_object(cls) for cls in CLASS_CENTERS}


@pytest.fixture(scope='session')
def synthetic_df():
    """12,000 synthetic rows. Shared by every test: copy it before changing it."""
    return make_synthetic_data()


@pytest.fixture(scope='session')
def tiny_model(synthetic_df):
    """The real pipeline (colors → final features → forest), small forest, fitted on synthetic data."""
    return build_model(**TINY_FOREST).fit(synthetic_df[FILTERS], synthetic_df[TARGET])


@pytest.fixture(scope='session')
def model_artifacts(tiny_model, synthetic_df, tmp_path_factory):
    """tiny_model saved by the real save code (src.artifacts) into a temp folder.

    Returns (model_path, metadata_path). The temp folder is created once and deleted by pytest.
    """
    folder = tmp_path_factory.mktemp('artifacts')
    model_path, metadata_path = folder / 'model.joblib', folder / 'metadata.json'

    preds = tiny_model.predict(synthetic_df[FILTERS])
    metadata = build_metadata(tiny_model, synthetic_df[FILTERS], synthetic_df[TARGET], preds,
                              data_version='synthetic')

    # save_artifacts writes to the paths in src.artifacts; point them at the temp folder for this call
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(src.artifacts, 'MODEL_PATH', model_path)
        mp.setattr(src.artifacts, 'METADATA_PATH', metadata_path)
        save_artifacts(tiny_model, metadata)
    return model_path, metadata_path


@pytest.fixture
def use_test_model(model_artifacts, monkeypatch):
    """Make src.predict (and so the API) load the temp test model instead of artifacts/.

    load_model caches what it read, so the cache is cleared before the test (drop anything loaded
    from elsewhere) and after it (don't leave the test model behind for the next test).
    """
    model_path, metadata_path = model_artifacts
    monkeypatch.setattr(src.predict, 'MODEL_PATH', model_path)
    monkeypatch.setattr(src.predict, 'METADATA_PATH', metadata_path)
    src.predict.load_model.cache_clear()
    yield model_path, metadata_path
    src.predict.load_model.cache_clear()
