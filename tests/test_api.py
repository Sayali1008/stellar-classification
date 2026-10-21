"""The HTTP API (app/main.py), called the way a browser or another program would.

TestClient sends requests to the app in memory: no server, no network, no Lambda.
"""
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from src.config import CLASSES, MAX_BATCH_SIZE, PLACEHOLDER


@pytest.fixture
def client(use_test_model):
    """A TestClient for the app, serving the temp test model.

    app.main loads the model as soon as it is imported, so it is imported here, after
    use_test_model has pointed src.predict at the test model. Importing it at the top of this file
    would load artifacts/ instead (missing in CI) before any fixture runs.
    """
    app_main = importlib.import_module('app.main')
    return TestClient(app_main.app)


def body(*objects):
    return {'objects': list(objects)}


# ---- GET /health ----

def test_health_reports_the_model_being_served(client, use_test_model):
    _, metadata_path = use_test_model
    expected_version = json.loads(metadata_path.read_text())['model_version']

    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'model_version': expected_version}


# ---- POST /predict ----

def test_predict_classifies_a_batch_in_order(client, class_centers):
    order = ['STAR', 'QSO', 'GALAXY']
    response = client.post('/predict', json=body(*(class_centers[c] for c in order)))
    assert response.status_code == 200

    result = response.json()
    assert result['model_version'] == client.get('/health').json()['model_version']
    assert [p['predicted_class'] for p in result['predictions']] == order
    for prediction in result['predictions']:
        assert set(prediction['probabilities']) == set(CLASSES)


def test_predict_explains_a_placeholder(client, class_centers):
    response = client.post('/predict', json=body({**class_centers['STAR'], 'u': PLACEHOLDER}))
    assert response.status_code == 422
    error = response.json()['detail'][0]
    # Where the problem is (first object, field u) and why, so the caller can fix it
    assert error['loc'] == ['body', 'objects', 0, 'u']
    assert 'placeholder' in error['msg']


@pytest.mark.parametrize('payload', [
    pytest.param(body(), id='no objects'),
    pytest.param({'objects': [{'u': 21.0}]}, id='missing magnitudes'),
    pytest.param({'objects': [{'u': 'abc', 'g': 20, 'r': 19, 'i': 19, 'z': 19}]}, id='text magnitude'),
    pytest.param({'stars': []}, id='wrong key'),
])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post('/predict', json=payload).status_code == 422


def test_predict_rejects_a_batch_over_the_limit(client, class_centers):
    payload = body(*[class_centers['STAR']] * (MAX_BATCH_SIZE + 1))
    assert client.post('/predict', json=payload).status_code == 422


def test_predict_rejects_a_body_that_is_not_json(client):
    response = client.post('/predict', content='not json', headers={'Content-Type': 'application/json'})
    assert response.status_code == 422


def test_predict_only_accepts_post(client):
    assert client.get('/predict').status_code == 405


# ---- The demo page and the docs ----

def test_index_serves_the_demo_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/html')
    assert 'Star, galaxy or quasar?' in response.text


def test_docs_page_loads(client):
    response = client.get('/docs')
    assert response.status_code == 200
    assert 'swagger' in response.text.lower()


def test_api_schema_lists_the_endpoints_but_not_the_page(client):
    paths = client.get('/openapi.json').json()['paths']
    assert set(paths) == {'/health', '/predict'}


def test_unknown_path_is_404(client):
    assert client.get('/nope').status_code == 404
