"""Inference (src/predict.py): loading the saved model and turning objects into predictions.

Runs against tiny_model, saved to a temp folder by the fixtures; use_test_model points
src.predict at it.
"""
import json
import subprocess
import sys

import joblib
import numpy as np
import pytest
import sklearn

import src.predict
from src.config import CLASSES, FILTERS, PROJECT_ROOT
from src.predict import load_model, model_version, predict
from src.schemas import StellarObject


def objects(dicts):
    return [StellarObject(**d) for d in dicts]


def test_saved_model_round_trips_through_joblib(model_artifacts, tiny_model, synthetic_df):
    # The file the API loads must give exactly the probabilities of the model that was trained
    model_path, _ = model_artifacts
    reloaded = joblib.load(model_path)
    sample = synthetic_df[FILTERS].head(200)
    np.testing.assert_array_equal(reloaded.predict(sample), tiny_model.predict(sample))
    # Not exactly equal: the forest adds up its trees on several CPU cores (n_jobs=-1), and the
    # order of that sum can change between calls, which moves the last of ~16 decimal digits
    np.testing.assert_allclose(reloaded.predict_proba(sample), tiny_model.predict_proba(sample), atol=1e-12)


def test_load_model_returns_the_model_and_its_metadata(use_test_model):
    model, metadata = load_model()
    assert sorted(model.classes_) == sorted(CLASSES)
    assert metadata['data_version'] == 'synthetic'
    assert model_version() == metadata['model_version']


def test_load_model_reads_the_files_only_once(use_test_model):
    # lru_cache: later calls return the same objects instead of reading the disk again
    assert load_model() is load_model()


def test_predicts_each_class_at_its_center(use_test_model, class_centers):
    preds = predict(objects(class_centers.values()))
    assert [p.predicted_class for p in preds] == list(class_centers)


def test_results_come_back_in_input_order(use_test_model, class_centers):
    order = ['QSO', 'STAR', 'GALAXY', 'STAR', 'QSO']
    preds = predict(objects(class_centers[c] for c in order))
    assert [p.predicted_class for p in preds] == order


def test_probabilities_are_one_per_class_and_sum_to_one(use_test_model, class_centers):
    for pred in predict(objects(class_centers.values())):
        assert set(pred.probabilities) == set(CLASSES)
        # Each probability is rounded to 3 decimals, so the sum can be off by a little
        assert sum(pred.probabilities.values()) == pytest.approx(1, abs=0.01)
        # The predicted class is the one with the highest probability
        assert pred.probabilities[pred.predicted_class] == max(pred.probabilities.values())


def test_metadata_records_each_filter_training_range(use_test_model):
    _, metadata = load_model()
    stats = metadata['input_stats']
    assert stats['range_quantiles'] == [0.01, 0.99]
    assert set(stats['ranges']) == set(FILTERS)
    for low, high in stats['ranges'].values():
        assert low < high


def test_typical_objects_get_no_warnings(use_test_model, class_centers):
    for pred in predict(objects(class_centers.values())):
        assert pred.warnings == []


def test_warns_once_per_filter_outside_the_training_range(use_test_model, class_centers):
    # Far brighter in u, far fainter in r: magnitudes work backwards
    extreme = {**class_centers['STAR'], 'u': 5.0, 'r': 40.0}
    [pred] = predict(objects([extreme]))
    assert len(pred.warnings) == 2
    assert pred.warnings[0].startswith('u = 5.00 is brighter than 99% of training objects')
    assert pred.warnings[1].startswith('r = 40.00 is fainter than 99% of training objects')
    # Still a normal prediction: the warning flags the answer, it doesn't replace it
    assert set(pred.probabilities) == set(CLASSES)


def test_model_without_input_stats_gets_no_warnings(use_test_model, monkeypatch, tmp_path, class_centers):
    # Models saved before input_stats existed (e.g. after a rollback) must still serve
    model_path, metadata_path = use_test_model
    old_metadata = json.loads(metadata_path.read_text())
    del old_metadata['input_stats']
    old_path = tmp_path / 'metadata.json'
    old_path.write_text(json.dumps(old_metadata))
    monkeypatch.setattr(src.predict, 'METADATA_PATH', old_path)
    load_model.cache_clear()

    [pred] = predict(objects([{**class_centers['STAR'], 'r': 40.0}]))
    assert pred.warnings == []


def test_empty_batch_returns_no_predictions(use_test_model):
    assert predict([]) == []


def test_refuses_a_model_saved_with_another_sklearn_version(use_test_model, monkeypatch):
    # Pretend a different scikit-learn is installed; the cache was cleared by use_test_model
    monkeypatch.setattr(sklearn, '__version__', '0.0.0')
    with pytest.raises(RuntimeError, match='saved with scikit-learn'):
        load_model()


def test_serving_code_loads_no_training_modules():
    # A fresh Python process, so modules imported by other tests don't count. The Lambda image
    # has none of these packages, so importing any of them from serving code would crash it.
    training_only = ['matplotlib', 'seaborn', 'boto3', 'mlflow', 'src.data', 'src.model', 'src.evaluate']
    code = f'import sys, src.predict; print([m for m in {training_only!r} if m in sys.modules])'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[]'
